#!/bin/bash
# Submit the 500-trial array, then queue the aggregation behind it.
#
#   cd SPO && bash slurm/run_context.sh
#
# Creates the venv from requirements.txt the first time; reuses it after that.
#
# The dependency is afterany, not afterok: if a handful of the 500 tasks die, the
# plots still get drawn from what did land and context_aggregate.py names the missing cells
# instead of quietly showing you a thinner boxplot.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results

module purge
module load gcc/13.3.0 python/3.11.9

VENV="$HOME/venvs/dfl"
if [ ! -d "$VENV" ]; then
    echo "creating venv at $VENV"
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV/bin/activate"
fi

trials_id=$(sbatch --parsable slurm/context_trials.sbatch)
echo "trials array submitted: ${trials_id}"

agg_id=$(sbatch --parsable --dependency="afterany:${trials_id}" slurm/context_aggregate.sbatch)
echo "aggregation submitted:  ${agg_id} (waits on ${trials_id})"

echo
echo "watch:   squeue -j ${trials_id},${agg_id}"
echo "rerun gaps: bash slurm/run_context.sh   # complete, current trials skip themselves"