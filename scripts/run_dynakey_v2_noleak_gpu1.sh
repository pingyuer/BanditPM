#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"

cd "${PROJECT_DIR}"
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

LOG_DIR="${LOG_DIR:-outputs/BanditPM/tmux_logs}"
mkdir -p "${LOG_DIR}"

run_exp() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "${UV_BIN}" run python train.py "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] END ${name}"
}

NOLEAK_ARGS=(
  --config-name config_dynakey_v2_noleak
  phase_init.train=pred_or_zero
  phase_init.val=pred_or_zero
  phase_init.test=pred_or_zero
  evaluation.init_mode=pred_or_zero
  evaluation.exclude_init_frame=true
  evaluation.init_frame_index=0
  evaluation.protocol_version=v2_no_leak
)

QLEARN_ARGS=(
  model.memory_core.dynakey.POLICY_MODE=q_greedy
  model.memory_core.dynakey.ENABLE_Q_LOSS=true
  model.memory_core.dynakey.LAMBDA_Q_CE=0.5
  model.memory_core.dynakey.LAMBDA_Q_ADV=0.05
  model.memory_core.dynakey.ADVANTAGE_CLAMP=2.0
  model.memory_core.dynakey.DETACH_Q_STATE=true
)

DYNAKEY_ARGS=(
  model.memory_core.dynakey.POLICY_MODE=fixed_residual
  model.memory_core.dynakey.ENABLE_Q_LOSS=false
)

run_exp camus_dynakey_qlearn_v2_noleak \
  "${NOLEAK_ARGS[@]}" \
  exp_id=camus_dynakey_qlearn_v2_noleak \
  wandb_mode=online \
  dataset_name=camus \
  data.protocol_name=camus_short_dense \
  data_path=/home/tahara/datasets/processed/camus_png256_10f \
  main_training.batch_size=8 \
  main_training.num_workers=8 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=dynakey_qlearn_v2_no_leak \
  wandb.tags='[camus,dynakey,kpff,ode_memory,q_learning,q_loss,v2_no_leak,predinit,exclude_init_frame]' \
  "${QLEARN_ARGS[@]}"

run_exp echonet_pediatric_a4c_full_cycle_dynakey_qlearn_v2_noleak \
  "${NOLEAK_ARGS[@]}" \
  exp_id=echonet_pediatric_a4c_full_cycle_dynakey_qlearn_v2_noleak \
  wandb_mode=online \
  dataset_name=echonet \
  data.protocol_name=echonet_pediatric_fullcycle_sparse \
  data_path=/home/tahara/datasets/processed/echonet_pediatric_a4c_full_cycle_png128_10f \
  main_training.batch_size=20 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=dynakey_qlearn_v2_no_leak \
  wandb.tags='[echonet_pediatric,a4c,full_cycle,dynakey,kpff,ode_memory,q_learning,q_loss,v2_no_leak,predinit,exclude_init_frame]' \
  "${QLEARN_ARGS[@]}"

run_exp cardiacuda_a4c_lv_dense_dynakey_qlearn_v2_noleak \
  "${NOLEAK_ARGS[@]}" \
  exp_id=cardiacuda_a4c_lv_dense_dynakey_qlearn_v2_noleak \
  wandb_mode=online \
  dataset_name=cardiacuda \
  data.protocol_name=cardiacuda_a4c_lv_dense \
  data_path=/home/tahara/datasets/processed/cardiacuda_a4c_lv_dense_png128_10f \
  main_training.batch_size=4 \
  main_training.num_workers=4 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=dynakey_qlearn_v2_no_leak \
  wandb.tags='[cardiacuda,a4c,lv,dense,dynakey,kpff,ode_memory,q_learning,q_loss,v2_no_leak,predinit,exclude_init_frame]' \
  "${QLEARN_ARGS[@]}"

run_exp camus_dynakey_v2_noleak \
  "${NOLEAK_ARGS[@]}" \
  exp_id=camus_dynakey_v2_noleak \
  wandb_mode=online \
  dataset_name=camus \
  data.protocol_name=camus_short_dense \
  data_path=/home/tahara/datasets/processed/camus_png256_10f \
  main_training.batch_size=8 \
  main_training.num_workers=8 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=dynakey_v2_no_leak \
  wandb.tags='[camus,dynakey,kpff,ode_memory,v2_no_leak,predinit,exclude_init_frame]' \
  "${DYNAKEY_ARGS[@]}"

run_exp echonet_pediatric_a4c_full_cycle_dynakey_v2_noleak \
  "${NOLEAK_ARGS[@]}" \
  exp_id=echonet_pediatric_a4c_full_cycle_dynakey_v2_noleak \
  wandb_mode=online \
  dataset_name=echonet \
  data.protocol_name=echonet_pediatric_fullcycle_sparse \
  data_path=/home/tahara/datasets/processed/echonet_pediatric_a4c_full_cycle_png128_10f \
  main_training.batch_size=20 \
  main_training.num_workers=12 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=dynakey_v2_no_leak \
  wandb.tags='[echonet_pediatric,a4c,full_cycle,dynakey,kpff,ode_memory,v2_no_leak,predinit,exclude_init_frame]' \
  "${DYNAKEY_ARGS[@]}"

run_exp cardiacuda_a4c_lv_dense_dynakey_v2_noleak \
  "${NOLEAK_ARGS[@]}" \
  exp_id=cardiacuda_a4c_lv_dense_dynakey_v2_noleak \
  wandb_mode=online \
  dataset_name=cardiacuda \
  data.protocol_name=cardiacuda_a4c_lv_dense \
  data_path=/home/tahara/datasets/processed/cardiacuda_a4c_lv_dense_png128_10f \
  main_training.batch_size=4 \
  main_training.num_workers=4 \
  eval_stage.eval_interval=200 \
  save=1 \
  save_weights_interval=500 \
  save_checkpoint_interval=0 \
  wandb.group=dynakey_v2_no_leak \
  wandb.tags='[cardiacuda,a4c,lv,dense,dynakey,kpff,ode_memory,v2_no_leak,predinit,exclude_init_frame]' \
  "${DYNAKEY_ARGS[@]}"
