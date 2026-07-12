#!/bin/bash
# Read every per-trial JSON and draw the boxplots. Cheap, single task -- normally
# submitted with a dependency on the trials array (see slurm/submit.sh), but you can
# also just run `python aggregate.py` on a login node once the array is done.

#SBATCH --partition=main
#SBATCH --account=guptavis_1305
#SBATCH --job-name=dfl-aggregate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# No Gurobi needed here: aggregate.py only touches numpy and matplotlib.
module purge
module load gcc/13.3.0 python/3.11.9
source "$HOME/venvs/dfl/bin/activate"

srun python aggregate.py --results results --outdir .