#!/usr/bin/env bash
set -euo pipefail

python ~/GDKVM/tools/preprocess_all.py \
  --input_root ~/datasets \
  --output_root ~/datasets/processed \
  --num_frames 10 \
  --num_visualizations 16 \
  --seed 42 \
  "$@"
