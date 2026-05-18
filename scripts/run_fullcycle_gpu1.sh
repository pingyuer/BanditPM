#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"

cd "${PROJECT_DIR}"
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

LOG_DIR=outputs/BanditPM/tmux_logs
mkdir -p "${LOG_DIR}"

run_exp() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "${UV_BIN}" run python train.py "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] END ${name}"
}

ECHONET_FC=${DATASETS_ROOT}/processed/echonet_full_cycle_png128_10f
PED_FC=${DATASETS_ROOT}/processed/echonet_pediatric_a4c_full_cycle_png128_10f

run_exp echonet_full_cycle_kpff \
  --config-name config_banditpm_baseline \
  exp_id=echonet_full_cycle_kpff \
  dataset_name=echonet \
  data_path=${ECHONET_FC} \
  main_training.batch_size=24 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  model.memory_core.type=none \
  model.temporal_memory.type=none \

run_exp echonet_full_cycle_bpm_rl \
  --config-name config_banditpm_baseline \
  exp_id=echonet_full_cycle_bpm_rl \
  dataset_name=echonet \
  data_path=${ECHONET_FC} \
  main_training.batch_size=20 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  model.memory_core.type=bpm \
  model.temporal_memory.type=bpm \
  model.temporal_memory.bpm.ENABLE=true \
  model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true \
  model.temporal_memory.bpm.USE_LEARNED_POLICY=true \
  model.temporal_memory.bpm.EXEC_POLICY=mixed \
  model.temporal_memory.bpm.ENABLE_POLICY_LOSS=true \
  model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=true \
  model.temporal_memory.bpm.ENABLE_RL_LOSS=true \

run_exp echonet_pediatric_a4c_full_cycle_gdkvm \
  --config-name config_banditpm_baseline \
  exp_id=echonet_pediatric_a4c_full_cycle_gdkvm \
  dataset_name=echonet \
  data_path=${PED_FC} \
  main_training.batch_size=24 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \

run_exp echonet_pediatric_a4c_full_cycle_bpm_rule \
  --config-name config_banditpm_baseline \
  exp_id=echonet_pediatric_a4c_full_cycle_bpm_rule \
  dataset_name=echonet \
  data_path=${PED_FC} \
  main_training.batch_size=20 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  model.memory_core.type=bpm \
  model.temporal_memory.type=bpm \
  model.temporal_memory.bpm.ENABLE=true \
  model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true \
  model.temporal_memory.bpm.USE_LEARNED_POLICY=false \
  model.temporal_memory.bpm.EXEC_POLICY=rule \
  model.temporal_memory.bpm.ENABLE_POLICY_LOSS=false \
  model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=false \
  model.temporal_memory.bpm.ENABLE_RL_LOSS=false \
