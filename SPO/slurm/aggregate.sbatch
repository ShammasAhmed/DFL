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

# Slurm runs this script from a spool copy, so resolve the real SPO/ first.
cd "$SLURM_SUBMIT_DIR"

# shellcheck source=slurm/env.sh
source slurm/env.sh
dfl_load_python
dfl_activate
dfl_threads

srun python aggregate.py --results results --outdir .