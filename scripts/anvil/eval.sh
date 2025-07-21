#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=ai250124
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:59:00
#SBATCH --job-name=eval
#SBATCH --output=/anvil/projects/ai250124/x-pkeung/anjo0/taudio/scripts/%x/%j.out
#SBATCH --error=/anvil/projects/ai250124/x-pkeung/anjo0/taudio/scripts/%x/%j.err

cd /anvil/projects/ai250124/x-pkeung/anjo0/taudio
module load conda
conda activate ./env
python evaluate.py --experiment $1 ${@:2}
