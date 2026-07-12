#!/bin/bash
# Run the whole sweep. This is the only command you need:
#
#   cd SPO && bash slurm/submit.sh
#
# It builds the venv from requirements.txt if there isn't one, imports the optimization
# stack once to prove it works, submits the 500-trial array, then queues the aggregation
# behind it. Re-running is cheap and safe: the venv is reused, finished trials skip
# themselves, so a second `bash slurm/submit.sh` just fills whatever gaps are left.
#
# Cluster-specific overrides (no file editing needed) -- see slurm/env.sh:
#   DFL_PYTHON_MODULE=python/3.11.3 DFL_ACCOUNT=abc_123 bash slurm/submit.sh
#
# The dependency below is afterany, not afterok: if a handful of the 500 tasks die, the
# plots still get drawn from what did land and aggregate.py names the missing cells
# instead of quietly showing you a thinner boxplot.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results

# shellcheck source=slurm/env.sh
source slurm/env.sh

dfl_load_python
dfl_bootstrap
dfl_activate
dfl_threads      # SLURM_CPUS_PER_TASK is unset here, so the preflight stays single-threaded
dfl_preflight

# Let DFL_ACCOUNT / DFL_PARTITION override the #SBATCH defaults baked into the .sbatch
# files, so a wrong account is a one-line fix rather than an edit in two places.
sb=()
[ -n "${DFL_ACCOUNT:-}" ]   && sb+=(--account="$DFL_ACCOUNT")
[ -n "${DFL_PARTITION:-}" ] && sb+=(--partition="$DFL_PARTITION")

trials_id=$(sbatch --parsable ${sb[@]+"${sb[@]}"} slurm/trials.sbatch)
echo "trials array submitted: ${trials_id}"

agg_id=$(sbatch --parsable ${sb[@]+"${sb[@]}"} \
    --dependency="afterany:${trials_id}" slurm/aggregate.sbatch)
echo "aggregation submitted:  ${agg_id} (waits on ${trials_id})"

echo
echo "watch:      squeue -j ${trials_id},${agg_id}"
echo "logs:       tail -f logs/trial_${trials_id}_0.out"
echo "rerun gaps: bash slurm/submit.sh   # finished trials skip themselves"