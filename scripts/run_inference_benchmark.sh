#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/benchmarks/inference_$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

PYTHONPATH=. "${UV_BIN}" run python scripts/benchmark_model_inference.py \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --frames "${FRAMES:-10}" \
  --size "${SIZE:-128}" \
  --warmup "${WARMUP:-5}" \
  --iters "${ITERS:-30}" \
  --amp \
  --output-dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "logs/inference_benchmark_$(date +%Y%m%d_%H%M%S).log"
