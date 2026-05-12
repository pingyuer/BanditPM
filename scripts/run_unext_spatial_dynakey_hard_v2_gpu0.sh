#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"

cd "${PROJECT_DIR}"
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

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
  evaluation.protocol_version=v2_no_leak_spatial_phase_dynakey_hard_v2
  wandb_mode=online
  save=1
  save_weights_interval=500
  save_checkpoint_interval=0
  eval_stage.eval_interval=200
  dataset_name=echonet
  data.protocol_name=echonet_ed2es_endpoint
  data_path=${DATASETS_ROOT}/processed/echonet_png128_10f
  main_training.batch_size=20
  main_training.num_workers=10
  wandb.group=unext_spatial_phase_dynakey_hard_v2
)

run_exp echonet_spatial_phase_full_hard_v2 config_unext_dynakey_spatial \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_spatial_phase_full_hard_v2 \
  wandb.tags='[echonet,spatial_phase,full,hard_v2,no_leak]'

run_exp echonet_spatial_phase_no_phase_hard_v2 config_unext_dynakey_spatial_no_phase \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_spatial_phase_no_phase_hard_v2 \
  wandb.tags='[echonet,spatial_phase,no_phase,hard_v2,no_leak]'

run_exp echonet_spatial_phase_broadcast_hard_v2 config_unext_dynakey_spatial_broadcast \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_spatial_phase_broadcast_hard_v2 \
  wandb.tags='[echonet,spatial_phase,broadcast_readout,hard_v2,no_leak]'

run_exp echonet_spatial_phase_no_dynamics_hard_v2 config_unext_dynakey_spatial_no_dynamics \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_spatial_phase_no_dynamics_hard_v2 \
  wandb.tags='[echonet,spatial_phase,no_dynamics,hard_v2,no_leak]'

run_exp echonet_spatial_phase_qdiag_hard_v2 config_unext_dynakey_spatial_q_diagnostic \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_spatial_phase_qdiag_hard_v2 \
  wandb.tags='[echonet,spatial_phase,q_diagnostic,hard_v2,no_leak]'

run_exp echonet_spatial_phase_qtrain_seg_hard_v2 config_unext_dynakey_spatial_q_training \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_spatial_phase_qtrain_seg_hard_v2 \
  wandb.tags='[echonet,spatial_phase,q_training,seg_reward,hard_v2,no_leak]'

run_exp echonet_spatial_phase_qtrain_latent_hard_v2 config_unext_dynakey_spatial_q_training_latent \
  "${COMMON_ARGS[@]}" \
  exp_id=echonet_spatial_phase_qtrain_latent_hard_v2 \
  wandb.tags='[echonet,spatial_phase,q_training,latent_reward,hard_v2,no_leak]'
