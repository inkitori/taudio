#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --job-name=train_and_eval
#SBATCH --output=/anvil/scratch/x-pkeung/anjo0/taudio/scripts/%x/%j.out
#SBATCH --error=/anvil/scratch/x-pkeung/anjo0/taudio/scripts/%x/%j.err

cd /anvil/scratch/x-pkeung/taudio
module load conda
conda activate ./env
python train.py --config $1
python evaluate.py --experiment $1 --split $2