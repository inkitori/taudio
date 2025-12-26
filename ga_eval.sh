#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <gpu_count> <config_path> <checkpoint_path> <split> [extra_args...]" >&2
  exit 1
fi

GPU_COUNT="$1"
CONFIG_PATH="$2"
CHECKPOINT_PATH="$3"
SPLIT="$4"

if [ "$SPLIT" == "test" ]; then
    EXTRA_FLAGS=""
elif [ "$SPLIT" == "dev" ]; then
    EXTRA_FLAGS="--dev"
else
    echo "Error: Invalid split '$SPLIT'. Must be 'test' or 'dev'."
    exit 1
fi

CMD=(
  accelerate launch
  --config_file "accelerate_configs/${GPU_COUNT}_gpu_bf16.yaml"
  ga_run.py
  --config "$CONFIG_PATH"
  --load-checkpoint "$CHECKPOINT_PATH"
  --eval-only
  $EXTRA_FLAGS
  "${@:5}"
)

echo "${CMD[@]}"
exec "${CMD[@]}"