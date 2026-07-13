#!/bin/bash
# One Slurm array task per trial of the histogram experiment.
#
#   2 sizes (100, 1000) x 500 trials = 1000 tasks, throttled to 40 running at once
#   (%40). Each task trains all 3 solvers (SPO+, LASSO, GBM) on its own draw, so the
#   array trains 3000 models against one fixed test context.
#
# Submit from the SPO/ directory -- normally via slurm/run_histogram.sh, which queues
# the aggregation behind this:
#   sbatch slurm/histogram_trials.sbatch
#
# Each task writes results_histogram/hist_n{n}_t{t}.json and skips itself if that file
# already exists and holds a decision for every solver the current code runs, so
# resubmitting after a failure only reruns the gaps.

#SBATCH --partition=main
#SBATCH --account=guptavis_1305
#SBATCH --job-name=dfl-histogram
#SBATCH --array=0-999%40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/histogram_%A_%a.out
#SBATCH --error=logs/histogram_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# No Gurobi module needed: the 5x5 grid is 40 variables, well inside the 2000-variable
# limit of the restricted license that ships with pip's gurobipy.
module purge
module load gcc/13.3.0 python/3.11.9
source "$HOME/venvs/dfl/bin/activate"

# Keep the numeric libraries inside the CPUs Slurm actually gave us; otherwise each of
# the 40 concurrent tasks tries to grab the whole node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --- Task ID -> (num_train, trial) -------------------------------------------
# Trial varies fastest, then training-set size. The values below must match
# HIST_SIZES / HIST_NUM_TRIALS in sweep.py, which histogram_trial.py validates its
# arguments against -- a mismatch here fails the task rather than silently
# mislabelling a cell.
SIZES=(100 1000)
NUM_TRIALS=500

T="$SLURM_ARRAY_TASK_ID"
trial=$(( T % NUM_TRIALS ))
n="${SIZES[$(( T / NUM_TRIALS ))]}"

echo "task ${T}: num_train=${n} trial=${trial} on $(hostname)"

srun python histogram_trial.py \
    --num-train "$n" \
    --trial "$trial" \
    --outdir results_histogram
