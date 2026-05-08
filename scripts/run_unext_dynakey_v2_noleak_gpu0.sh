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
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "${UV_BIN}" run python train.py "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] END ${name}"
}

COMMON_ARGS=(
  --config-name config_unext_dynakey
  phase_init.train=pred_or_zero
  phase_init.val=pred_or_zero
  phase_init.test=pred_or_zero
  evaluation.init_mode=pred_or_zero
  evaluation.exclude_init_frame=true
  evaluation.init_frame_index=0
  evaluation.protocol_version=v2_no_leak_unext_dynakey
  wandb_mode=online
  save=1
  save_weights_interval=500
  save_checkpoint_interval=0
  eval_stage.eval_interval=200
  model.unext_dynakey.use_temporal_refine=true
  model.memory_core.dynakey.POLICY_MODE=fixed_residual
  model.memory_core.dynakey.ENABLE_Q_LOSS=false
)

run_exp echonet_unext_dynakey_v2_noleak \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_unext_dynakey_v2_noleak \
  dataset_name=echonet \
  data.protocol_name=echonet_ed2es_endpoint \
  data_path=${DATASETS_ROOT}/processed/echonet_png128_10f \
  main_training.batch_size=24 \
  main_training.num_workers=12 \
  wandb.group=unext_dynakey_v2_no_leak \
  wandb.tags='[echonet,unext,dynakey,ode_memory,temporal_refine,v2_no_leak,predinit,exclude_init_frame]'

run_exp echonet_full_cycle_unext_dynakey_v2_noleak \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_full_cycle_unext_dynakey_v2_noleak \
  dataset_name=echonet \
  data.protocol_name=echonet_fullcycle_sparse \
  data_path=${DATASETS_ROOT}/processed/echonet_full_cycle_png128_10f \
  main_training.batch_size=24 \
  main_training.num_workers=12 \
  wandb.group=unext_dynakey_v2_no_leak \
  wandb.tags='[echonet,full_cycle,unext,dynakey,ode_memory,temporal_refine,v2_no_leak,predinit,exclude_init_frame]'

run_exp cardiacuda_a4c_lv_sparse_unext_dynakey_v2_noleak \
  "${COMMON_ARGS[@]}" \
  exp_id=cardiacuda_a4c_lv_sparse_unext_dynakey_v2_noleak \
  dataset_name=cardiacuda \
  data.protocol_name=cardiacuda_a4c_lv_sparse \
  data_path=${DATASETS_ROOT}/processed/cardiacuda_a4c_lv_png128_10f \
  main_training.batch_size=16 \
  main_training.num_workers=8 \
  wandb.group=unext_dynakey_v2_no_leak \
  wandb.tags='[cardiacuda,a4c,lv,sparse,unext,dynakey,ode_memory,temporal_refine,v2_no_leak,predinit,exclude_init_frame]'
