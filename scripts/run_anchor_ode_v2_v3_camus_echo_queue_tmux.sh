#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
SESSION_NAME="${SESSION_NAME:-anchor_ode_v2_v3_queue}"
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
  run_one 0 anchor_ode_v2_v3_echo_e2_long_raw_tta anchor_ode_v2_v3_echo_e2_long_raw_tta echonet echonet_ed2es_endpoint "${DATASETS_ROOT}/processed/echonet_png128_10f" 20 10 "[canonical,anchor_ode_v2_v3,echo,e2_long_raw_tta,current_anchor_affine,hflip_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v3_echo_e2_long_raw_tta.log"
  run_one 0 anchor_ode_v2_v3_echo_boundaryless_long anchor_ode_v2_v3_echo_boundaryless_long echonet echonet_ed2es_endpoint "${DATASETS_ROOT}/processed/echonet_png128_10f" 20 10 "[canonical,anchor_ode_v2_v3,echo,boundaryless_long,current_anchor_affine,warp_delta,hflip_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v3_echo_boundaryless_long.log"
  run_one 0 anchor_ode_v2_v3_echo_capacity_mild anchor_ode_v2_v3_echo_capacity_mild echonet echonet_ed2es_endpoint "${DATASETS_ROOT}/processed/echonet_png128_10f" 20 10 "[canonical,anchor_ode_v2_v3,echo,capacity_mild,current_anchor_affine,hflip_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v3_echo_capacity_mild.log"
}

run_camus_queue() {
  run_one 1 anchor_ode_v2_v3_camus_skip_only_tta anchor_ode_v2_v3_camus_skip_only_tta camus camus_short_dense "${DATASETS_ROOT}/processed/camus_png256_10f" 8 8 "[canonical,anchor_ode_v2_v3,camus,skip_only_tta,current_anchor_affine,hflip_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v3_camus_skip_only_tta.log"
  run_one 1 anchor_ode_v2_v3_camus_skip_only_long anchor_ode_v2_v3_camus_skip_only_long camus camus_short_dense "${DATASETS_ROOT}/processed/camus_png256_10f" 8 8 "[canonical,anchor_ode_v2_v3,camus,skip_only_long,current_anchor_affine,hflip_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v3_camus_skip_only_long.log"
  run_one 1 anchor_ode_v2_v3_camus_skip_only_capacity anchor_ode_v2_v3_camus_skip_only_capacity camus camus_short_dense "${DATASETS_ROOT}/processed/camus_png256_10f" 8 8 "[canonical,anchor_ode_v2_v3,camus,skip_only_capacity,current_anchor_affine,hflip_tta,postprocess]" 2>&1 | tee "${LOG_DIR}/anchor_ode_v2_v3_camus_skip_only_capacity.log"
}

tmux new-session -d -s "${SESSION_NAME}" -n echo "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START echo v3 queue' && $(declare -f run_one run_echo_queue); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; LOG_DIR='${LOG_DIR}'; run_echo_queue; echo '[\$(date +%F\\ %T)] END echo v3 queue'; exec bash"
tmux new-window -t "${SESSION_NAME}" -n camus "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START camus v3 queue' && $(declare -f run_one run_camus_queue); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; LOG_DIR='${LOG_DIR}'; run_camus_queue; echo '[\$(date +%F\\ %T)] END camus v3 queue'; exec bash"

echo "Started tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Logs are under: ${PROJECT_DIR}/${LOG_DIR}/anchor_ode_v2_v3_*.log"
