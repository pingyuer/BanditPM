#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
METHOD="${METHOD:-all}"
DATASET="${DATASET:-all}"

cd "${PROJECT_DIR}"
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATASETS_ROOT

LOG_DIR="${LOG_DIR:-outputs/BanditPM/tmux_logs}"
mkdir -p "${LOG_DIR}"
EXTRA_ARGS=("$@")

METHODS=(gdkvm kpff unext_fusion delay_ode)
DATASETS=(echo camus domain)

select_values() {
  local requested="$1"
  shift
  local values=("$@")
  if [[ "${requested}" == "all" ]]; then
    printf '%s\n' "${values[@]}"
    return
  fi
  for value in "${values[@]}"; do
    if [[ "${value}" == "${requested}" ]]; then
      printf '%s\n' "${value}"
      return
    fi
  done
  echo "Unknown selection '${requested}'. Available: all ${values[*]}" >&2
  return 2
}

run_exp() {
  local name="$1"
  echo "[$(date '+%F %T')] START ${name}"
  "${UV_BIN}" run python train.py --config-name "${name}" wandb_mode="${WANDB_MODE:-online}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] END ${name}"
}

mapfile -t selected_methods < <(select_values "${METHOD}" "${METHODS[@]}")
mapfile -t selected_datasets < <(select_values "${DATASET}" "${DATASETS[@]}")

for method in "${selected_methods[@]}"; do
  for dataset in "${selected_datasets[@]}"; do
    run_exp "${method}_${dataset}"
  done
done
