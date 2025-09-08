#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --job-name=eval
#SBATCH --output=/anvil/scratch/x-pkeung/taudio/scripts/logs/%x/%j.out
#SBATCH --error=/anvil/scratch/x-pkeung/taudio/scripts/logs/%x/%j.err

cd /anvil/scratch/x-pkeung/taudio
module load conda
conda activate ./env

# Build the evaluate command with optional arguments
eval_cmd="python evaluate.py --experiment $1 --split $2"

if [ -n "$3" ]; then
    eval_cmd="$eval_cmd --error-bound $3"
fi

if [ -n "$4" ]; then
    eval_cmd="$eval_cmd --min-time $4"
fi

if [ -n "$5" ]; then
    eval_cmd="$eval_cmd --max-time $5"
fi

$eval_cmd
