#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_path> [checkpoint_path]" >&2
  exit 1
fi

CONFIG_PATH="$1"
CHECKPOINT_PATH="${2-}"

CMD=(
  accelerate launch
  --config_file accelerate_configs/4_gpu_bf16.yaml
  run.py
  --config "$CONFIG_PATH"
)

if [[ -n "$CHECKPOINT_PATH" ]]; then
  CMD+=(--load-checkpoint "$CHECKPOINT_PATH")
fi

exec "${CMD[@]}"