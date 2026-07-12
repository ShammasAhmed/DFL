#!/bin/bash
# Build the venv, without submitting anything. Run this on a LOGIN node:
#
#   cd SPO && bash slurm/setup.sh
#
# You don't normally need this -- slurm/submit.sh does the same bootstrap before it
# submits. It exists for the case where you want an interactive node to time a trial
# before committing 500 tasks to the queue, since compute nodes have no internet and
# can't build the venv themselves.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results

# shellcheck source=slurm/env.sh
source slurm/env.sh

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "WARNING: you appear to be inside a Slurm allocation (job $SLURM_JOB_ID)." >&2
    echo "         Compute nodes usually have no outbound internet; if pip hangs or" >&2
    echo "         fails to reach PyPI, exit and rerun this on a login node." >&2
fi

dfl_load_python
dfl_bootstrap
dfl_activate
dfl_threads
dfl_preflight

echo
echo "venv ready: $DFL_VENV"
echo
echo "interactive use:  source slurm/activate.sh"
echo "submit the sweep: bash slurm/submit.sh"