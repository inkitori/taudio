#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_path> [checkpoint_path]" >&2
  exit 1
fi

CONFIG_PATH="$1"
CHECKPOINT_PATH="${2-}"

if [ "$3" == "test" ]; then
    EXTRA_FLAGS=""
elif [ "$3" == "dev" ]; then
    EXTRA_FLAGS="--dev"
else
    echo "Error: Invalid split '$3'. Must be 'test' or 'dev'."
    exit 1
fi

CMD=(
  accelerate launch
  --config_file accelerate_configs/4_gpu_bf16.yaml
  run.py
  --config "$1"
  --load-checkpoint "$2"
  --eval-only
  $EXTRA_FLAGS
)

echo $CMD

exec "${CMD[@]}"