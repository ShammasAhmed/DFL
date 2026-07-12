#!/bin/bash
# Submit the 500-trial array, then queue the aggregation behind it.
#
#   cd SPO && bash slurm/submit.sh
#
# The dependency is afterany, not afterok: if a handful of the 500 tasks die, the
# plots still get drawn from what did land and aggregate.py names the missing cells
# instead of quietly showing you a thinner boxplot.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results

trials_id=$(sbatch --parsable slurm/trials.sbatch)
echo "trials array submitted: ${trials_id}"

agg_id=$(sbatch --parsable --dependency="afterany:${trials_id}" slurm/aggregate.sbatch)
echo "aggregation submitted:  ${agg_id} (waits on ${trials_id})"

echo
echo "watch:   squeue -j ${trials_id},${agg_id}"
echo "rerun gaps: sbatch slurm/trials.sbatch   # finished trials skip themselves"