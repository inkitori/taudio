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

# TAKE_FIRST=""
# if [ -n "$4" ]; then
# TAKE_FIRST="--take-first $4"
# fi

CMD=(
  accelerate launch
  --config_file accelerate_configs/1_gpu_bf16.yaml
  ga_run.py
  --config "$1"
  --load-checkpoint "$2"
  --eval-only
  $EXTRA_FLAGS
  # $TAKE_FIRST
)

echo $CMD

exec "${CMD[@]}"