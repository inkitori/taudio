#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --job-name=4_gpu_bf16
#SBATCH --output=scripts/anvil/logs/%x/%j.out
#SBATCH --error=scripts/anvil/logs/%x/%j.err

echo "-------------------------------------------------------"
echo "mybalance:"
mybalance
echo "-------------------------------------------------------"

start_time=$(date +%s)

export OMP_NUM_THREADS=$(lscpu -b -p=CPU | grep -v '^#' | wc -l)

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR: $MASTER_ADDR"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT: $MASTER_PORT"

module load conda
conda activate $SCRATCH/envs/taudio


# Optional eval min/max time arguments
EVAL_MIN_ARG=""
if [ -n "$2" ]; then
EVAL_MIN_ARG="--eval-min-time $2"
fi

EVAL_MAX_ARG=""
if [ -n "$3" ]; then
EVAL_MAX_ARG="--eval-max-time $3"
fi

# Check if $1 contains "qwen3b" and set batch size argument accordingly
# BS_ARG=""
# if [[ "$1" == *"qwen3b"* ]]; then
#     BS_ARG="--bs-per-device 2"
#     echo "Config contains 'qwen3b': Setting --bs-per-device 2"
# fi


accelerate launch --config_file accelerate_configs/4_gpu_bf16.yaml ga_run.py --config "$1" $EVAL_MIN_ARG $EVAL_MAX_ARG # $BS_ARG

# --- END TIMING ---
end_time=$(date +%s)
runtime=$((end_time - start_time))

# Format the seconds into Hours:Minutes:Seconds
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$((runtime % 60))

echo "-------------------------------------------------------"
echo "Job Execution Completed"
echo "Total Runtime: $(printf "%02d:%02d:%02d" $hours $minutes $seconds)"
echo "mybalance:"
mybalance
echo "-------------------------------------------------------"