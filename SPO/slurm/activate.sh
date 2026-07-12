# Source me (don't run me) to get a working shell for interactive runs:
#
#   cd SPO
#   source slurm/activate.sh
#   python run_trial.py --deg 8 --num-train 1000 --trial 0
#
# Loads the Python module and activates the venv. On a login node it will build the
# venv if there isn't one; inside a Slurm allocation it refuses to, because compute
# nodes generally have no outbound internet and a pip install there would either hang
# or fail halfway. Bootstrap on the login node first.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "activate.sh must be sourced, not executed:" >&2
    echo "    source slurm/activate.sh" >&2
    exit 1
fi

# shellcheck source=slurm/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

dfl_load_python

if [ ! -x "$DFL_VENV/bin/python" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "[env] ERROR: no venv at $DFL_VENV, and this is a compute node" >&2
        echo "[env]        (job $SLURM_JOB_ID). Compute nodes have no internet, so pip" >&2
        echo "[env]        would fail here. Build it on a login node first:" >&2
        echo "[env]            cd SPO && bash slurm/setup.sh" >&2
    else
        dfl_bootstrap
    fi
fi

dfl_activate && dfl_threads && echo "[env] ready: $(python --version 2>&1), venv $DFL_VENV"