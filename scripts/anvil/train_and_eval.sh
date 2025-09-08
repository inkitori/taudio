#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --job-name=train_and_eval
#SBATCH --output=/anvil/scratch/x-pkeung/taudio/scripts/logs/%x/%j.out
#SBATCH --error=/anvil/scratch/x-pkeung/taudio/scripts/logs/%x/%j.err

cd /anvil/scratch/x-pkeung/taudio
module load conda
conda activate ./env
# Capture the training output to extract the experiment directory
train_output=$(python train.py --config $1 2>&1) # this also has the effect of piping all train to out, and eval to err
echo "$train_output"

# Extract the experiment directory from training output
experiment_dir=$(echo "$train_output" | grep "Training completed. All outputs saved to:" | sed 's/.*All outputs saved to: //')

if [ -z "$experiment_dir" ]; then
    echo "ERROR: Could not extract experiment directory from training output"
    exit 1
fi

# Extract just the experiment name (last part of the path)
experiment_name=$(basename "$experiment_dir")

echo "Training completed, starting evaluation..."
echo "Experiment directory: $experiment_dir"
echo "Experiment name: $experiment_name"

# Build the evaluate command with the exact experiment name
eval_cmd="python evaluate.py --experiment $experiment_name --split $2"

if [ -n "$3" ]; then
    eval_cmd="$eval_cmd --error-bound $3"
fi

if [ -n "$4" ]; then
    eval_cmd="$eval_cmd --min-time $4"
fi

if [ -n "$5" ]; then
    eval_cmd="$eval_cmd --max-time $5"
fi

echo "Eval command: $eval_cmd"
$eval_cmd