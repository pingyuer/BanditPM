#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
SESSION_NAME="${SESSION_NAME:-anchor_ode_v2_v4_queue}"
LOG_DIR="${LOG_DIR:-outputs/BanditPM/tmux_logs}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not available."
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  fallback="${SESSION_NAME}_$(date +%Y%m%d_%H%M%S)"
  echo "tmux session '${SESSION_NAME}' already exists; using '${fallback}'."
  SESSION_NAME="${fallback}"
fi

COMMON_ARGS=(
  phase_init.train=pred_or_zero
  phase_init.val=pred_or_zero
  phase_init.test=pred_or_zero
  evaluation.init_mode=pred_or_zero
  evaluation.exclude_init_frame=true
  evaluation.init_frame_index=0
  evaluation.protocol_version=v3_current_anchor_affine
  save=1
  save_weights_interval=500
  save_checkpoint_interval=0
  eval_stage.eval_interval=200
)

run_one() {
  local gpu="$1"
  local name="$2"
  local config_name="$3"
  local dataset_name="$4"
  local protocol_name="$5"
  local data_path="$6"
  local batch_size="$7"
  local workers="$8"
  local tags="$9"

  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=. \
    HYDRA_FULL_ERROR=1 \
    DATASETS_ROOT="${DATASETS_ROOT}" \
    "${UV_BIN}" run python train.py \
    --config-name "${config_name}" \
    "${COMMON_ARGS[@]}" \
    dataset_name="${dataset_name}" \
    data.protocol_name="${protocol_name}" \
    data_path="${data_path}" \
    main_training.batch_size="${batch_size}" \
    main_training.num_workers="${workers}" \
    exp_id="${name}" \
    "hydra.run.dir=outputs/BanditPM/${name}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
}

run_echo_queue() {
  run_one 0 anchor_ode_v2_v4_echo_e2_raw_fine anchor_ode_v2_v4_echo_e2_raw_fine echonet echonet_ed2es_endpoint "${DATASETS_ROOT}/processed/echonet_png128_10f" 20 10 "[canonical,anchor_ode_v2_v4,echo,e2_raw_fine,current_anchor_affine,no_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v4_echo_e2_raw_fine.log"
  run_one 0 anchor_ode_v2_v4_echo_base_guard anchor_ode_v2_v4_echo_base_guard echonet echonet_ed2es_endpoint "${DATASETS_ROOT}/processed/echonet_png128_10f" 20 10 "[canonical,anchor_ode_v2_v4,echo,base_guard,current_anchor_affine,no_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v4_echo_base_guard.log"
  run_one 0 anchor_ode_v2_v4_echo_skip_only_raw anchor_ode_v2_v4_echo_skip_only_raw echonet echonet_ed2es_endpoint "${DATASETS_ROOT}/processed/echonet_png128_10f" 20 10 "[canonical,anchor_ode_v2_v4,echo,skip_only_raw,current_anchor_affine,no_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v4_echo_skip_only_raw.log"
}

run_camus_queue() {
  run_one 1 anchor_ode_v2_v4_camus_skip_long_repro anchor_ode_v2_v4_camus_skip_long_repro camus camus_short_dense "${DATASETS_ROOT}/processed/camus_png256_10f" 8 8 "[canonical,anchor_ode_v2_v4,camus,skip_long_repro,current_anchor_affine,hflip_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v4_camus_skip_long_repro.log"
  run_one 1 anchor_ode_v2_v4_camus_early_sched anchor_ode_v2_v4_camus_early_sched camus camus_short_dense "${DATASETS_ROOT}/processed/camus_png256_10f" 8 8 "[canonical,anchor_ode_v2_v4,camus,early_sched,current_anchor_affine,hflip_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v4_camus_early_sched.log"
  run_one 1 anchor_ode_v2_v4_camus_trust_guided anchor_ode_v2_v4_camus_trust_guided camus camus_short_dense "${DATASETS_ROOT}/processed/camus_png256_10f" 8 8 "[canonical,anchor_ode_v2_v4,camus,trust_guided,current_anchor_affine,hflip_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v4_camus_trust_guided.log"
}

tmux new-session -d -s "${SESSION_NAME}" -n echo "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START echo v4 queue' && $(declare -f run_one run_echo_queue); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; LOG_DIR='${LOG_DIR}'; run_echo_queue; echo '[\$(date +%F\\ %T)] END echo v4 queue'; exec bash"
tmux new-window -t "${SESSION_NAME}" -n camus "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START camus v4 queue' && $(declare -f run_one run_camus_queue); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; LOG_DIR='${LOG_DIR}'; run_camus_queue; echo '[\$(date +%F\\ %T)] END camus v4 queue'; exec bash"

echo "Started tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Logs are under: ${PROJECT_DIR}/${LOG_DIR}/anchor_ode_v2_v4_*.log"
