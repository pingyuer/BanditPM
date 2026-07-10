#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
BASE_PORT="${BASE_PORT:-29541}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/fair_12_queue_${STAMP}}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  gdkvm_echo_fair_endpoint10
  dpfr_echo_fair_endpoint10
  gdkvm_echonet_pediatric_fair_endpoint10
  dpfr_echonet_pediatric_fair_endpoint10
  gdkvm_camus_fair_dense10
  dpfr_camus_fair_dense10
  gdkvm_cardiacuda_g2r_fair
  dpfr_cardiacuda_g2r_fair
  gdkvm_cardiacuda_r2g_fair
  dpfr_cardiacuda_r2g_fair
  gdkvm_cardiacuda_sparse_sitegen_fair
  dpfr_cardiacuda_sparse_sitegen_fair
)

{
  echo -e "index\tconfig\tstatus\tstarted_at\tfinished_at\tlog"
} > "${LOG_DIR}/manifest.tsv"

echo "[queue] root=${ROOT_DIR}"
echo "[queue] log_dir=${LOG_DIR}"
echo "[queue] gpus=${GPUS} nproc=${NPROC}"
echo "[queue] MLflow is required; preflight is enabled for every run."

for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  run_no=$((idx + 1))
  port=$((BASE_PORT + idx))
  run_log="${LOG_DIR}/${run_no}_${config}.log"
  started_at="$(date --iso-8601=seconds)"
  echo "[queue] (${run_no}/${#CONFIGS[@]}) start ${config} at ${started_at}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  PYTHONPATH=src:. \
  HYDRA_FULL_ERROR=1 \
  "${UV_BIN}" run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port="${port}" \
    train.py \
    --config-name "${config}" \
    smoke_skip_eval=false \
    mlflow.enabled=true \
    mlflow.required=true \
    mlflow.artifacts_enabled=true \
    mlflow.artifacts_required=true \
    mlflow.preflight=true \
    > "${run_log}" 2>&1
  rc=$?
  set -e
  finished_at="$(date --iso-8601=seconds)"
  if [[ "${rc}" -eq 0 ]]; then
    status="finished"
    echo "[queue] (${run_no}/${#CONFIGS[@]}) finished ${config} at ${finished_at}"
  else
    status="failed_${rc}"
    echo "[queue] (${run_no}/${#CONFIGS[@]}) FAILED ${config} rc=${rc} at ${finished_at}"
    echo -e "${run_no}\t${config}\t${status}\t${started_at}\t${finished_at}\t${run_log}" >> "${LOG_DIR}/manifest.tsv"
    echo "[queue] stopping after failure; inspect ${run_log}"
    exit "${rc}"
  fi
  echo -e "${run_no}\t${config}\t${status}\t${started_at}\t${finished_at}\t${run_log}" >> "${LOG_DIR}/manifest.tsv"
done

echo "[queue] all ${#CONFIGS[@]} runs finished"
