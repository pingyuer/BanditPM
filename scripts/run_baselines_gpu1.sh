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

run_exp echonet_kpff \
  --config-name config_banditpm_baseline \
  exp_id=echonet_kpff \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_png128_10f \
  main_training.batch_size=24 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  model.memory_core.type=none \
  model.temporal_memory.type=none \
  wandb.group=echonet_baselines \
  wandb.tags='[echonet,kpff,no_gdr]'

run_exp echonet_bpm_rl \
  --config-name config_banditpm_baseline \
  exp_id=echonet_bpm_rl \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_png128_10f \
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
  wandb.group=echonet_baselines \
  wandb.tags='[echonet,bpm_rl,kpff,bpm,rl]'

run_exp camus_gdkvm \
  --config-name config_banditpm_baseline \
  exp_id=camus_gdkvm \
  wandb_mode=online \
  dataset_name=camus \
  data_path=/home/tahara/datasets/processed/camus_png256_10f \
  main_training.batch_size=10 \
  main_training.num_workers=8 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=camus_baselines \
  wandb.tags='[camus,gdkvm,kpff,gdr]'

run_exp camus_bpm_rule \
  --config-name config_banditpm_baseline \
  exp_id=camus_bpm_rule \
  wandb_mode=online \
  dataset_name=camus \
  data_path=/home/tahara/datasets/processed/camus_png256_10f \
  main_training.batch_size=8 \
  main_training.num_workers=8 \
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
  wandb.group=camus_baselines \
  wandb.tags='[camus,bpm_rule,kpff,bpm]'
