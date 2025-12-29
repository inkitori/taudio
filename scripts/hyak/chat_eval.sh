#!/bin/bash
#SBATCH --job-name=chat_eval
#SBATCH --account=ark
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:a40:1
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

WANDB_RUN_ARG=""
if [ -n "$3" ]; then
WANDB_RUN_ARG="--resume-wandb-run-id $3"
fi

python evaluate_api_timestamp_any.py --datasets $1 --chatgpt-api-key $2 --log-wandb --providers chatgpt $WANDB_RUN_ARG