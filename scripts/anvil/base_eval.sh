#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --job-name=base_eval
#SBATCH --output=scripts/anvil/logs/%x/%j.out
#SBATCH --error=scripts/anvil/logs/%x/%j.err

module load conda
conda activate ./env
python evaluate_base.py --model-id $1 --repository $2 --split $3 --task $4