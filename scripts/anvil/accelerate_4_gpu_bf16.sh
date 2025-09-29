#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=1:30:00
#SBATCH --job-name=4_gpu_bf16
#SBATCH --output=scripts/anvil/logs/%x/%j.out
#SBATCH --error=scripts/anvil/logs/%x/%j.err

export OMP_NUM_THREADS=$(lscpu -b -p=CPU | grep -v '^#' | wc -l)

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR: $MASTER_ADDR"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT: $MASTER_PORT"

module load conda
conda activate ./env
# Capture the training output to extract the experiment directory
train_output=$(accelerate launch --config_file accelerate_configs/4_gpu_bf16.yaml accelerate_train.py --config $1 2>&1) # this also has the effect of piping all train to out, and eval to err
echo "$train_output"

# Extract the experiment directory from training output
experiment_dir=$(echo "$train_output" | grep "Training completed. All outputs saved to:" | sed 's/.*All outputs saved to: //')

if [ -z "$experiment_dir" ]; then
    echo "ERROR: Could not extract experiment directory from training output"
    exit 1
fi

echo "Training completed, starting evaluation job"
echo "Experiment directory: $experiment_dir"

eval_cmd="sbatch scripts/anvil/eval.sh $experiment_dir $2"

if [ -n "$3" ]; then
    eval_cmd="$eval_cmd $3"
fi

if [ -n "$4" ]; then
    eval_cmd="$eval_cmd $4"
fi

echo "Eval command: $eval_cmd"
$eval_cmd