#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"

cd "${PROJECT_DIR}"
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR="${LOG_DIR:-outputs/BanditPM/tmux_logs}"
mkdir -p "${LOG_DIR}"

run_exp() {
  local name="$1"
  local config_name="$2"
  shift 2
  echo "[$(date '+%F %T')] START ${name}"
  "${UV_BIN}" run python train.py --config-name "${config_name}" "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] END ${name}"
}

COMMON_ARGS=(
  phase_init.train=pred_or_zero
  phase_init.val=pred_or_zero
  phase_init.test=pred_or_zero
  evaluation.init_mode=pred_or_zero
  evaluation.exclude_init_frame=true
  evaluation.init_frame_index=0
  save=1
  save_weights_interval=500
  save_checkpoint_interval=0
  eval_stage.eval_interval=200
  dataset_name=echonet
  data.protocol_name=echonet_ed2es_endpoint
  data_path=${DATASETS_ROOT}/processed/echonet_png128_10f
  main_training.batch_size=24
  main_training.num_workers=12
)

run_exp echonet_unext_only_v2_noleak config_unext_only \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_unext_only_v2_noleak

run_exp echonet_unext_temporal_refine_v2_noleak config_unext_temporal_refine \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_unext_temporal_refine_v2_noleak

run_exp echonet_unext_dynakey_no_refine_v2_noleak config_unext_dynakey_no_refine \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_unext_dynakey_no_refine_v2_noleak

run_exp echonet_unext_dynakey_full_v2_noleak config_unext_dynakey \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_unext_dynakey_full_v2_noleak
