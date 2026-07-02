#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <gpu_id> <config_name> [config_name ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
CONFIGS=("$@")

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-$(command -v uv)}"
TS="${FAIR_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${PROJECT_DIR}/logs/fair_debel_gdkvm_${TS}_gpu${GPU_ID}"
MLFLOW_URI="${MLFLOW_URI:-http://172.16.240.77:5000}"
WAIT_PID="${WAIT_PID:-}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[fair-queue] Waiting for PID ${WAIT_PID} before using GPU ${GPU_ID}."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 300
  done
fi

for CONFIG in "${CONFIGS[@]}"; do
  LOG_FILE="${LOG_DIR}/${CONFIG}.log"
  RUN_DIR="outputs/fair_debel_gdkvm/${TS}/${CONFIG}/\${now:%H-%M-%S}"
  echo "[fair-queue] GPU=${GPU_ID} CONFIG=${CONFIG} LOG=${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${UV_BIN}" run python train.py \
    --config-name "${CONFIG}" \
    mlflow.tracking_uri="${MLFLOW_URI}" \
    hydra.run.dir="${RUN_DIR}" \
    2>&1 | tee "${LOG_FILE}"
done

echo "[fair-queue] Done GPU=${GPU_ID}"
