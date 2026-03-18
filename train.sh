#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(($RANDOM % 40000 + 20000))

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1

export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

IFS=',' read -ra _cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#_cuda_devices[@]}

echo "🚀 Starting training on ${NUM_GPUS} GPUs (IDs: ${CUDA_VISIBLE_DEVICES}) on Port ${MASTER_PORT}..."
echo "📂 Project Root: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [ -x "${PROJECT_ROOT}/.venv/bin/torchrun" ]; then
  TORCHRUN_BIN="${PROJECT_ROOT}/.venv/bin/torchrun"
  TORCHRUN_PREFIX=()
elif command -v uv >/dev/null 2>&1; then
  TORCHRUN_BIN="torchrun"
  TORCHRUN_PREFIX=(uv run)
elif command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_BIN="torchrun"
  TORCHRUN_PREFIX=()
else
  echo "Neither .venv/bin/torchrun, uv, nor torchrun is available."
  exit 1
fi

GPU_CHECK_OUTPUT="$("${PROJECT_ROOT}/.venv/bin/python" - <<'PY'
import sys
import torch

print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"cuda:{idx}={torch.cuda.get_device_name(idx)}")
else:
    sys.exit(2)
PY
)" || {
  echo "CUDA check failed before launch."
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "${PROJECT_ROOT}/.venv/bin/python" - <<'PY'
import os
import torch

print(f"torch={torch.__version__}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
PY
  exit 1
}

echo "${GPU_CHECK_OUTPUT}"

"${TORCHRUN_PREFIX[@]}" \
  "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node=${NUM_GPUS} \
  train.py \
  "$@" \
  hydra.run.dir=outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}
