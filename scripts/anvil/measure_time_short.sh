#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --job-name=measure_time
#SBATCH --output=scripts/anvil/logs/%x/%j.out
#SBATCH --error=scripts/anvil/logs/%x/%j.err

module load conda
conda activate $SCRATCH/envs/taudio

python measure_time.py --checkpoint $1 --config $2 --batch-size $3