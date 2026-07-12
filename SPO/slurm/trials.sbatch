#!/bin/bash
# One Slurm array task per trial of the (degree x training-set-size) sweep.
#
#   5 degrees (1,2,4,6,8) x 2 sizes (100,1000) x 50 trials = 500 tasks,
#   throttled to 25 running at once (%25).
#
# Submit from the SPO/ directory:
#   mkdir -p logs results
#   sbatch slurm/trials.sbatch
#
# Each task writes results/deg{d}_n{n}_t{t}.json and skips itself if that file
# already exists, so resubmitting after a failure only reruns the gaps.

#SBATCH --partition=main
#SBATCH --account=guptavis_1305
#SBATCH --job-name=dfl-trials
#SBATCH --array=0-499%25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/trial_%A_%a.out
#SBATCH --error=logs/trial_%A_%a.err

set -euo pipefail

# Slurm runs this script from a spool copy, so resolve the real SPO/ first.
cd "$SLURM_SUBMIT_DIR"

# Python module, venv, and thread pinning all live in one place.
# No Gurobi module needed: the 5x5 grid is 40 edge variables, comfortably inside the
# 2000-variable limit of the restricted license bundled with pip's gurobipy.
# shellcheck source=slurm/env.sh
source slurm/env.sh
dfl_load_python
dfl_activate
dfl_threads

# --- Task ID -> (deg, num_train, trial) --------------------------------------
# Must match sweep.cells(): trial varies fastest, then degree, then size.
DEGREES=(1 2 4 6 8)
SIZES=(100 1000)
NUM_TRIALS=50

T="$SLURM_ARRAY_TASK_ID"
trial=$(( T % NUM_TRIALS ))
cfg=$(( T / NUM_TRIALS ))
deg="${DEGREES[$(( cfg % ${#DEGREES[@]} ))]}"
n="${SIZES[$(( cfg / ${#DEGREES[@]} ))]}"

echo "task ${T}: deg=${deg} num_train=${n} trial=${trial} on $(hostname)"

srun python run_trial.py \
    --deg "$deg" \
    --num-train "$n" \
    --trial "$trial" \
    --num-contexts 1000 \
    --outdir results