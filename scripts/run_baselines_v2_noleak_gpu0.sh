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

NOLEAK_ARGS=(
  --config-name config_banditpm_baseline
  phase_init.train=pred_or_zero
  phase_init.val=pred_or_zero
  phase_init.test=pred_or_zero
  evaluation.init_mode=pred_or_zero
  evaluation.exclude_init_frame=true
  evaluation.init_frame_index=0
  evaluation.protocol_version=v2_no_leak
  save=1
  save_weights_interval=500
  save_checkpoint_interval=0
)

run_gdkvm() {
  local name="$1" dataset="$2" path="$3" protocol="$4" batch="$5" workers="$6" tags="$7"
  run_exp "${name}_gdkvm_v2_noleak" "${NOLEAK_ARGS[@]}" \
    exp_id="${name}_gdkvm_v2_noleak" dataset_name="${dataset}" data_path="${path}" data.protocol_name="${protocol}" \
    main_training.batch_size="${batch}" main_training.num_workers="${workers}" eval_stage.eval_interval=200
}

run_kpff() {
  local name="$1" dataset="$2" path="$3" protocol="$4" batch="$5" workers="$6" tags="$7"
  run_exp "${name}_kpff_v2_noleak" "${NOLEAK_ARGS[@]}" \
    exp_id="${name}_kpff_v2_noleak" dataset_name="${dataset}" data_path="${path}" data.protocol_name="${protocol}" \
    main_training.batch_size="${batch}" main_training.num_workers="${workers}" eval_stage.eval_interval=200 \
    model.memory_core.type=none model.temporal_memory.type=none
}

run_bpm_rule() {
  local name="$1" dataset="$2" path="$3" protocol="$4" batch="$5" workers="$6" tags="$7"
  run_exp "${name}_bpm_rule_v2_noleak" "${NOLEAK_ARGS[@]}" \
    exp_id="${name}_bpm_rule_v2_noleak" dataset_name="${dataset}" data_path="${path}" data.protocol_name="${protocol}" \
    main_training.batch_size="${batch}" main_training.num_workers="${workers}" eval_stage.eval_interval=200 \
    model.memory_core.type=bpm model.temporal_memory.type=bpm \
    model.temporal_memory.bpm.ENABLE=true \
    model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true \
    model.temporal_memory.bpm.USE_LEARNED_POLICY=false \
    model.temporal_memory.bpm.EXEC_POLICY=rule \
    model.temporal_memory.bpm.ENABLE_POLICY_LOSS=false \
    model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=false \
    model.temporal_memory.bpm.ENABLE_RL_LOSS=false
}

run_bpm_rl() {
  local name="$1" dataset="$2" path="$3" protocol="$4" batch="$5" workers="$6" tags="$7"
  run_exp "${name}_bpm_rl_v2_noleak" "${NOLEAK_ARGS[@]}" \
    exp_id="${name}_bpm_rl_v2_noleak" dataset_name="${dataset}" data_path="${path}" data.protocol_name="${protocol}" \
    main_training.batch_size="${batch}" main_training.num_workers="${workers}" eval_stage.eval_interval=200 \
    model.memory_core.type=bpm model.temporal_memory.type=bpm \
    model.temporal_memory.bpm.ENABLE=true \
    model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true \
    model.temporal_memory.bpm.USE_LEARNED_POLICY=true \
    model.temporal_memory.bpm.EXEC_POLICY=mixed \
    model.temporal_memory.bpm.ENABLE_POLICY_LOSS=true \
    model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=true \
    model.temporal_memory.bpm.ENABLE_RL_LOSS=true
}

run_all_methods() {
  local name="$1" dataset="$2" path="$3" protocol="$4" batch_fast="$5" batch_bpm="$6" workers="$7" tags="$8"
  run_gdkvm "${name}" "${dataset}" "${path}" "${protocol}" "${batch_fast}" "${workers}" "${tags}"
  run_kpff "${name}" "${dataset}" "${path}" "${protocol}" "${batch_fast}" "${workers}" "${tags}"
  run_bpm_rule "${name}" "${dataset}" "${path}" "${protocol}" "${batch_bpm}" "${workers}" "${tags}"
  run_bpm_rl "${name}" "${dataset}" "${path}" "${protocol}" "${batch_bpm}" "${workers}" "${tags}"
}

run_all_methods echonet echonet ${DATASETS_ROOT}/processed/echonet_png128_10f echonet_ed2es_endpoint 24 20 12 "[echonet"
run_all_methods echonet_full_cycle echonet ${DATASETS_ROOT}/processed/echonet_full_cycle_png128_10f echonet_fullcycle_sparse 24 20 12 "[echonet,full_cycle"
run_all_methods cardiacuda_a4c_lv_sparse cardiacuda ${DATASETS_ROOT}/processed/cardiacuda_a4c_lv_png128_10f cardiacuda_a4c_lv_sparse 16 16 8 "[cardiacuda,a4c,lv,sparse"
