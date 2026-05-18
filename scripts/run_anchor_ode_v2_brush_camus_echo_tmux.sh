#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
SESSION_NAME="${SESSION_NAME:-anchor_ode_v2_brush_camus_echo}"
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

tmux new-session -d -s "${SESSION_NAME}" -n echo "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START echo brush' && $(declare -f run_one); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; LOG_DIR='${LOG_DIR}'; run_one 0 anchor_ode_v2_brush_echo anchor_ode_v2_brush_echo echonet echonet_ed2es_endpoint '${DATASETS_ROOT}/processed/echonet_png128_10f' 20 10 '[canonical,anchor_ode_v2_brush,current_anchor_affine,echo,ema,hflip_tta,postprocess]' 2>&1 | tee '${LOG_DIR}/anchor_ode_v2_brush_echo.log'; echo '[\$(date +%F\\ %T)] END echo brush'; exec bash"
tmux new-window -t "${SESSION_NAME}" -n camus "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START camus brush' && $(declare -f run_one); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; LOG_DIR='${LOG_DIR}'; run_one 1 anchor_ode_v2_brush_camus anchor_ode_v2_brush_camus camus camus_short_dense '${DATASETS_ROOT}/processed/camus_png256_10f' 8 8 '[canonical,anchor_ode_v2_brush,current_anchor_affine,camus,ema,hflip_tta,postprocess]' 2>&1 | tee '${LOG_DIR}/anchor_ode_v2_brush_camus.log'; echo '[\$(date +%F\\ %T)] END camus brush'; exec bash"

echo "Started tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Logs are under: ${PROJECT_DIR}/${LOG_DIR}/anchor_ode_v2_brush_*.log"
