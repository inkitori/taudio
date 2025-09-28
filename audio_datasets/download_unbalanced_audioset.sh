#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --job-name=download_unbalanced_audioset
#SBATCH --output=audio_datasets/logs/%x/%j.out
#SBATCH --error=audio_datasets/logs/%x/%j.err

cd audio_datasets
module load conda
conda activate ./dataset_env

python download_unbalanced_audioset.py