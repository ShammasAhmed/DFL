#!/bin/bash
# THE HISTOGRAM EXPERIMENT. Submit the 1000-trial array, then queue the plotting
# behind it.
#
#   cd SPO && bash slurm/run_histogram.sh
#
# 500 trials x {n=100, n=1000} x 3 solvers = 3000 models trained, all evaluated on one
# fixed test context, counted into a histogram of how often each solver picks a path of
# each rank.
#
# Creates the venv from requirements.txt the first time; reuses it after that.
#
# The dependency is afterany, not afterok: if a handful of the 1000 tasks die, the
# histograms still get drawn from what did land and histogram_aggregate.py names the
# missing trials instead of quietly showing you a thinner bar.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results_histogram

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

trials_id=$(sbatch --parsable slurm/histogram_trials.sbatch)
echo "histogram trials array submitted: ${trials_id}"

agg_id=$(sbatch --parsable --dependency="afterany:${trials_id}" \
    slurm/histogram_aggregate.sbatch)
echo "histogram plotting submitted:     ${agg_id} (waits on ${trials_id})"

echo
echo "watch:      squeue -j ${trials_id},${agg_id}"
echo "output:     path_histogram_n{100,1000}_{gbm,lasso,spo,compare}.png"
echo "            path_histogram_counts.csv"
echo "rerun gaps: bash slurm/run_histogram.sh   # complete trials skip themselves"
