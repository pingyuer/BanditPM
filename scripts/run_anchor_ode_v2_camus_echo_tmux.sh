#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
SESSION_NAME="${SESSION_NAME:-anchor_ode_v2_camus_echo}"
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

echo_cmd() {
  local gpu="$1"
  local name="$2"
  local config_name="$3"
  shift 3

  printf '%q ' \
    env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=. \
    HYDRA_FULL_ERROR=1 \
    DATASETS_ROOT="${DATASETS_ROOT}" \
    "${UV_BIN}" run python train.py \
    --config-name "${config_name}" \
    "$@" \
    "exp_id=${name}" \
    "hydra.run.dir=outputs/BanditPM/${name}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
}

ECHO_CMD="$(echo_cmd 0 anchor_ode_v2_echo anchor_ode_v2_echo \
  "${COMMON_ARGS[@]}" \
  dataset_name=echonet \
  data.protocol_name=echonet_ed2es_endpoint \
  "data_path=${DATASETS_ROOT}/processed/echonet_png128_10f" \
  main_training.batch_size=20 \
  main_training.num_workers=10)"

CAMUS_CMD="$(echo_cmd 1 anchor_ode_v2_camus anchor_ode_v2_camus \
  "${COMMON_ARGS[@]}" \
  dataset_name=camus \
  data.protocol_name=camus_short_dense \
  "data_path=${DATASETS_ROOT}/processed/camus_png256_10f" \
  main_training.batch_size=8 \
  main_training.num_workers=8)"

tmux new-session -d -s "${SESSION_NAME}" -n echo "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START anchor_ode_v2_echo' && ${ECHO_CMD} 2>&1 | tee '${LOG_DIR}/anchor_ode_v2_echo.log'; echo '[\$(date +%F\\ %T)] END anchor_ode_v2_echo'; exec bash"
tmux new-window -t "${SESSION_NAME}" -n camus "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START anchor_ode_v2_camus' && ${CAMUS_CMD} 2>&1 | tee '${LOG_DIR}/anchor_ode_v2_camus.log'; echo '[\$(date +%F\\ %T)] END anchor_ode_v2_camus'; exec bash"

echo "Started tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Logs:"
echo "  ${PROJECT_DIR}/${LOG_DIR}/anchor_ode_v2_echo.log"
echo "  ${PROJECT_DIR}/${LOG_DIR}/anchor_ode_v2_camus.log"
