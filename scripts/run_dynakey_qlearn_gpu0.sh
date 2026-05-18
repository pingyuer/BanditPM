#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"

cd "${PROJECT_DIR}"
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

LOG_DIR=outputs/BanditPM/tmux_logs
mkdir -p "${LOG_DIR}"

run_exp() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "${UV_BIN}" run python train.py "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] END ${name}"
}

QLEARN_ARGS=(
  model.memory_core.dynakey.POLICY_MODE=q_greedy
  model.memory_core.dynakey.ENABLE_Q_LOSS=true
  model.memory_core.dynakey.LAMBDA_Q_CE=0.5
  model.memory_core.dynakey.LAMBDA_Q_ADV=0.05
  model.memory_core.dynakey.ADVANTAGE_CLAMP=2.0
  model.memory_core.dynakey.DETACH_Q_STATE=true
)

run_exp echonet_dynakey_qlearn \
  --config-name config_dynakey_baseline \
  exp_id=echonet_dynakey_qlearn \
  dataset_name=echonet \
  data_path=${DATASETS_ROOT}/processed/echonet_png128_10f \
  main_training.batch_size=20 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  "${QLEARN_ARGS[@]}"

run_exp echonet_full_cycle_dynakey_qlearn \
  --config-name config_dynakey_baseline \
  exp_id=echonet_full_cycle_dynakey_qlearn \
  dataset_name=echonet \
  data_path=${DATASETS_ROOT}/processed/echonet_full_cycle_png128_10f \
  main_training.batch_size=20 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  "${QLEARN_ARGS[@]}"

run_exp cardiacuda_a4c_lv_sparse_oracle_dynakey_qlearn \
  --config-name config_dynakey_baseline \
  exp_id=cardiacuda_a4c_lv_sparse_oracle_dynakey_qlearn \
  dataset_name=cardiacuda \
  data_path=${DATASETS_ROOT}/processed/cardiacuda_a4c_lv_png128_10f \
  main_training.batch_size=16 \
  main_training.num_workers=8 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  "${QLEARN_ARGS[@]}"
