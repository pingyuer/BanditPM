#!/usr/bin/env bash
set -euo pipefail

cd /home/tahara/GDKVM
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

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

DYNAKEY_ARGS=(
  model.memory_core.type=dynakey
  model.temporal_memory.type=dynakey
  +model.memory_core.dynakey.BANK_SIZE=4
  +model.memory_core.dynakey.DT=1.0
  +model.memory_core.dynakey.EMA_ALPHA=0.2
  +model.memory_core.dynakey.RETRIEVAL_TEMPERATURE=1.0
  +model.memory_core.dynakey.HIDDEN_DIM=256
  +model.memory_core.dynakey.GATE_INIT=1.0
)

run_exp echonet_dynakey \
  --config-name config_banditpm_baseline \
  exp_id=echonet_dynakey \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_png128_10f \
  main_training.batch_size=20 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=echonet_dynakey \
  wandb.tags='[echonet,dynakey,kpff,ode_memory]' \
  "${DYNAKEY_ARGS[@]}"

run_exp echonet_full_cycle_dynakey \
  --config-name config_banditpm_baseline \
  exp_id=echonet_full_cycle_dynakey \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_full_cycle_png128_10f \
  main_training.batch_size=20 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=echonet_fullcycle_dynakey \
  wandb.tags='[echonet,full_cycle,dynakey,kpff,ode_memory]' \
  "${DYNAKEY_ARGS[@]}"

run_exp cardiacuda_a4c_lv_sparse_oracle_dynakey \
  --config-name cardiacuda_a4c_lv_oracle \
  exp_id=cardiacuda_a4c_lv_sparse_oracle_dynakey \
  wandb_mode=online \
  dataset_name=cardiacuda \
  data_path=/home/tahara/datasets/processed/cardiacuda_a4c_lv_png128_10f \
  main_training.batch_size=16 \
  main_training.num_workers=8 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=cardiacuda_dynakey \
  wandb.tags='[cardiacuda,a4c,lv,sparse,oracle,dynakey,kpff,ode_memory]' \
  "${DYNAKEY_ARGS[@]}"
