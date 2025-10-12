#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --job-name=eval
#SBATCH --output=scripts/anvil/logs/%x/%j.out
#SBATCH --error=scripts/anvil/logs/%x/%j.err

module load conda
conda activate ./env

# Build the evaluate command with optional arguments
eval_cmd="python evaluate.py --experiment $1 --split $2" # --k-errs"

if [ -n "$3" ]; then
    eval_cmd="$eval_cmd --min-time $3"
fi

if [ -n "$4" ]; then
    eval_cmd="$eval_cmd --max-time $4"
fi

$eval_cmd
