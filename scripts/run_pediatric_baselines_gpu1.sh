#!/usr/bin/env bash
set -euo pipefail

cd /home/tahara/GDKVM
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

UV_BIN=/home/tahara/miniconda3/bin/uv
LOG_DIR=outputs/BanditPM/tmux_logs
mkdir -p "${LOG_DIR}"

run_exp() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "${UV_BIN}" run python train.py "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] END ${name}"
}

DATA_PATH=/home/tahara/datasets/processed/echonet_pediatric_a4c_png128_10f

run_exp echonet_pediatric_a4c_kpff \
  --config-name config_banditpm_baseline \
  exp_id=echonet_pediatric_a4c_kpff \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=${DATA_PATH} \
  main_training.batch_size=24 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  model.memory_core.type=none \
  model.temporal_memory.type=none \
  wandb.group=echonet_pediatric_a4c_baselines \
  wandb.tags='[echonet_pediatric,a4c,kpff,no_gdr]'

run_exp echonet_pediatric_a4c_bpm_rl \
  --config-name config_banditpm_baseline \
  exp_id=echonet_pediatric_a4c_bpm_rl \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=${DATA_PATH} \
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
  wandb.group=echonet_pediatric_a4c_baselines \
  wandb.tags='[echonet_pediatric,a4c,bpm_rl,kpff,bpm,rl]'
