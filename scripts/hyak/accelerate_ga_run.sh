#!/bin/bash
#SBATCH --job-name=gemini_eval
#SBATCH --account=ark
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:a40:4
#SBATCH --output=scripts/%x/%j.out
#SBATCH --error=scripts/%x/%j.err

CONDA_BASE=$(conda info --base) # This is a good way to get it if conda is in PATH

echo "CONDA_BASE detected as: ${CONDA_BASE}" # For debugging

# Source the conda.sh script
# The exact path might vary slightly based on your Conda version / installation type
# but etc/profile.d/conda.sh is standard
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    echo "Sourced ${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi

cd /gscratch/ark/anjo0/taudio
conda activate taudio

# Optional eval min/max time arguments
EVAL_MIN_ARG=""
if [ -n "$2" ]; then
EVAL_MIN_ARG="--eval-min-time $2"
fi

EVAL_MAX_ARG=""
if [ -n "$3" ]; then
EVAL_MAX_ARG="--eval-max-time $3"
fi

accelerate launch --config_file accelerate_configs/4_gpu_bf16.yaml run.py --config "$1" $EVAL_MIN_ARG $EVAL_MAX_ARG