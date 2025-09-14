#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=1:59:00
#SBATCH --job-name=accelerate_train_2_gpu
#SBATCH --output=/anvil/scratch/x-pkeung/taudio/scripts/logs/%x/%j.out
#SBATCH --error=/anvil/scratch/x-pkeung/taudio/scripts/logs/%x/%j.err

cd /anvil/scratch/x-pkeung/taudio
module load conda
conda activate ./env

accelerate launch --config_file $2 accelerate_train.py --config $1