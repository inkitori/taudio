#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=nairr250124-ai
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
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

accelerate launch --config_file accelerate_configs/4_gpu_bf16.yaml run.py --config $1 