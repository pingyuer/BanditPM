#!/bin/zsh
set -euo pipefail

PROJECT_ROOT=${0:a:h}

export CUDA_VISIBLE_DEVICES=0,1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(($RANDOM % 40000 + 20000))

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1

export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

typeset -a _cuda_devices
_cuda_devices=(${(s:,:)CUDA_VISIBLE_DEVICES})
NUM_GPUS=${#_cuda_devices[@]}

echo "🚀 Starting training on ${NUM_GPUS} GPUs (IDs: ${CUDA_VISIBLE_DEVICES}) on Port ${MASTER_PORT}..."
echo "📂 Project Root: ${PROJECT_ROOT}"

# 
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x "${PROJECT_ROOT}/.venv/bin/torchrun" ]]; then
  TORCHRUN_CMD=("${PROJECT_ROOT}/.venv/bin/torchrun")
elif command -v uv >/dev/null 2>&1; then
  TORCHRUN_CMD=(uv run torchrun)
elif command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_CMD=(torchrun)
else
  echo "Neither .venv/bin/torchrun, uv, nor torchrun is available."
  exit 1
fi

"${TORCHRUN_CMD[@]}" \
  --standalone \
  --nproc_per_node=${NUM_GPUS} \
  train.py \
  "$@" \
  hydra.run.dir=outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}
