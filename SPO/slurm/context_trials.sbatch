#!/bin/bash
# One Slurm array task per trial of the (degree x training-set-size) sweep.
#
#   5 degrees (1,2,4,6,8) x 2 sizes (100,1000) x 50 trials = 500 tasks,
#   throttled to 25 running at once (%25).
#
# Submit from the SPO/ directory:
#   mkdir -p logs results
#   sbatch slurm/context_trials.sbatch
#
# Each task writes results/deg{d}_n{n}_h{H}_t{t}.json and skips itself if that file already
# exists AND holds every metric the current code computes, so resubmitting after a
# failure only reruns the gaps -- while a results/ left over from before a metric was
# added is recomputed rather than silently kept.

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

cd "$SLURM_SUBMIT_DIR"

# No Gurobi module needed: the 5x5 grid is 40 variables, well inside the 2000-variable
# limit of the restricted license that ships with pip's gurobipy.
module purge
module load gcc/13.3.0 python/3.11.9
source "$HOME/venvs/dfl/bin/activate"

# Keep the numeric libraries inside the CPUs Slurm actually gave us; otherwise each of
# the 25 concurrent tasks tries to grab the whole node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --- Task ID -> (deg, num_train, trial) --------------------------------------
# Trial varies fastest, then degree, then training-set size. The values below must
# match DEGREES / SIZES / NUM_TRIALS in sweep.py, which context_trial.py validates its
# arguments against -- a mismatch here fails the task rather than silently
# mislabelling a cell.
DEGREES=(1 2 4 6 8)
SIZES=(100 1000)
NUM_TRIALS=50

# Multiplicative noise half-width (epsilon ~ U[1-H, 1+H]), shared by every task in the
# array. It is recorded in each result JSON and in the filename, so a run at a different
# H can share results/ without colliding. context_aggregate.py reads one H at a time
# (its --h, default 0.5), so aggregate with --h "$H" to match.
H=0.01

T="$SLURM_ARRAY_TASK_ID"
trial=$(( T % NUM_TRIALS ))
cfg=$(( T / NUM_TRIALS ))
deg="${DEGREES[$(( cfg % ${#DEGREES[@]} ))]}"
n="${SIZES[$(( cfg / ${#DEGREES[@]} ))]}"

echo "task ${T}: deg=${deg} num_train=${n} trial=${trial} h=${H} on $(hostname)"

srun python context_trial.py \
    --deg "$deg" \
    --num-train "$n" \
    --trial "$trial" \
    --noise-width "$H" \
    --num-contexts 1000 \
    --outdir results