#!/bin/bash
# Shared environment for every piece of the Slurm pipeline.
#
# Sourced by submit.sh (login node) and by trials.sbatch / aggregate.sbatch (compute
# nodes), so all three agree on which Python and which venv they use. Sourcing this
# only defines DFL_* paths and dfl_* functions -- nothing runs on its own.
#
# Override any of these from the command line instead of editing a file:
#
#   DFL_PYTHON_MODULE=python/3.11.3   module to load (default: first candidate that works)
#   DFL_VENV=$HOME/venvs/dfl          where the venv lives (default: SPO/.venv)
#   DFL_ACCOUNT=abc_123               Slurm account   (default: the one in the .sbatch files)
#   DFL_PARTITION=epyc-64             Slurm partition (default: the one in the .sbatch files)
#
#   e.g.  DFL_PYTHON_MODULE=python/3.11.3 bash slurm/submit.sh

# The SPO/ directory, regardless of where this file was sourced from.
DFL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DFL_VENV="${DFL_VENV:-$DFL_ROOT/.venv}"

# All 500 tasks import matplotlib (run_trial -> main -> plots). Without this they race
# to build the font cache in $HOME on first run. One shared cache instead, warmed once
# by the preflight on the login node.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$DFL_ROOT/.cache/matplotlib}"

# Tried in order; the first one that loads wins. An entry may name several modules.
#
# python/3.11.9 is pinned deliberately -- it matches the interpreter requirements.txt
# was pinned against. Do NOT relax this to a bare "python": on USC CARC the default
# module is 3.13.2, which is not what these versions were resolved for.
#
# The gcc/13.3.0 pair comes first because on CARC the Spack tree holding python/3.11.9
# is only on MODULEPATH while gcc/13.3.0 is loaded, and `module purge` drops it. The
# bare entry is the fallback for the flat /apps/lmod/modules/utils tree.
DFL_PYTHON_CANDIDATES=(
    "${DFL_PYTHON_MODULE:-}"
    "gcc/13.3.0 python/3.11.9"
    "python/3.11.9"
    "python/3.11"
)

# requirements.txt was resolved against this interpreter; warn loudly on anything else.
DFL_PYTHON_EXPECT="3.11"


dfl_load_python() {
    if ! command -v module >/dev/null 2>&1; then
        echo "[env] no module system found; using python3 from PATH"
    else
        module purge >/dev/null 2>&1 || true
        local m loaded=""
        for m in "${DFL_PYTHON_CANDIDATES[@]}"; do
            [ -n "$m" ] || continue
            # $m is intentionally unquoted: an entry may name more than one module.
            # shellcheck disable=SC2086
            if module load $m >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
                echo "[env] module load $m"
                loaded="$m"
                break
            fi
            module purge >/dev/null 2>&1 || true
        done
        if [ -z "$loaded" ]; then
            echo "[env] WARNING: none of these loaded: ${DFL_PYTHON_CANDIDATES[*]}" >&2
            echo "[env]          falling back to python3 on PATH. Run 'module avail python'" >&2
            echo "[env]          and set DFL_PYTHON_MODULE=<name> to fix." >&2
        fi
    fi

    # A 3.13 interpreter will happily blow up on the pins in requirements.txt.
    local have
    have="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")"
    if [ "$have" != "$DFL_PYTHON_EXPECT" ]; then
        echo "[env] WARNING: python $have, but requirements.txt is pinned for $DFL_PYTHON_EXPECT." >&2
        echo "[env]          Set DFL_PYTHON_MODULE to a $DFL_PYTHON_EXPECT module if pip fails." >&2
    else
        echo "[env] python $have"
    fi
}


# Create the venv and install requirements.txt. No-op if the venv already exists,
# so this is safe to call on every submit.
dfl_bootstrap() {
    if [ -x "$DFL_VENV/bin/python" ]; then
        echo "[env] venv already present: $DFL_VENV"
        return 0
    fi

    local py
    if   command -v python3 >/dev/null 2>&1; then py=python3
    elif command -v python  >/dev/null 2>&1; then py=python
    else echo "[env] ERROR: no python on PATH." >&2; return 1
    fi

    echo "[env] creating venv at $DFL_VENV ($("$py" --version 2>&1))"
    "$py" -m venv "$DFL_VENV"
    dfl_activate

    python -m pip install --quiet --upgrade pip setuptools wheel

    # CPU-only torch first, from PyTorch's own index. If that index doesn't carry this
    # version for the cluster's platform, fall through and let requirements.txt pull the
    # default wheel -- bigger, but it still runs.
    local torch_pin
    torch_pin="$(grep -E '^torch==' "$DFL_ROOT/requirements.txt" | head -1)"
    echo "[env] installing $torch_pin (CPU wheel)"
    python -m pip install --index-url https://download.pytorch.org/whl/cpu "$torch_pin" \
        || echo "[env] WARNING: no CPU wheel for $torch_pin; falling back to the default index" >&2

    echo "[env] installing requirements.txt"
    python -m pip install -r "$DFL_ROOT/requirements.txt"
    echo "[env] venv ready"
}


dfl_activate() {
    if [ ! -x "$DFL_VENV/bin/python" ]; then
        echo "[env] ERROR: no venv at $DFL_VENV." >&2
        echo "[env]        Run 'bash slurm/submit.sh' on a login node -- it builds one." >&2
        return 1
    fi
    # dfl_bootstrap already activates the venv it just built; don't stack PATH twice.
    [ "${VIRTUAL_ENV:-}" = "$DFL_VENV" ] && return 0
    # The activate script trips over `set -u` on older virtualenvs.
    set +u
    # shellcheck disable=SC1091
    source "$DFL_VENV/bin/activate"
    set -u
}


# Keep the numeric libraries inside the CPUs Slurm actually gave us; otherwise each of
# the concurrent array tasks tries to grab the whole node.
dfl_threads() {
    local n="${SLURM_CPUS_PER_TASK:-1}"
    export OMP_NUM_THREADS="$n"
    export MKL_NUM_THREADS="$n"
    export OPENBLAS_NUM_THREADS="$n"
    export NUMEXPR_NUM_THREADS="$n"
}


# Import the real thing before committing 500 tasks to the queue: this builds the
# gurobipy model (so a missing/oversized license fails here, on the login node, once)
# and warms the matplotlib font cache.
dfl_preflight() {
    echo "[env] preflight..."
    ( cd "$DFL_ROOT" && python - <<'PY'
import matplotlib
matplotlib.use("Agg")
from main import optmodel, SOLVERS
print(f"[env] preflight ok: grid={optmodel.grid}, solvers={[k for k, _ in SOLVERS]}")
PY
    )
}