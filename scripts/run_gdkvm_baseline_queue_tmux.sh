#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
SESSION_NAME="${SESSION_NAME:-gdkvm_baseline_queue}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-anchor_ode_v2_v4_queue}"
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
  evaluation.protocol_version=v3_canonical_no_leak
  mlflow.stage=full
  mlflow.required=true
  mlflow.artifacts_required=true
  mlflow.preflight=false
  save=1
  save_weights_interval=500
  save_checkpoint_interval=0
  eval_stage.eval_interval=500
  eval_stage.final_eval=true
  eval_stage.final_test=true
  eval_stage.test_every_eval=false
  eval_stage.test_interval=0
  eval_stage.num_vis=0
  evaluation.threshold_search_during_training=false
  evaluation.threshold_search_start=0.30
  evaluation.threshold_search_end=0.75
  evaluation.threshold_search_step=0.01
)

wait_for_session() {
  local session="$1"
  if [ -z "${session}" ]; then
    return 0
  fi
  while tmux has-session -t "${session}" 2>/dev/null; do
    echo "[$(date +%F\ %T)] waiting for tmux session '${session}' to finish..."
    sleep 300
  done
}

run_one() {
  local gpu="$1"
  local name="$2"
  local config_name="$3"
  local dataset_name="$4"
  local protocol_name="$5"
  local data_path="$6"
  local batch_size="$7"
  local workers="$8"
  local command_log_path="$9"

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
    mlflow.command_log_path="${command_log_path}" \
    exp_id="${name}" \
    "hydra.run.dir=outputs/BanditPM/${name}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
}

run_echo_queue() {
  wait_for_session "${WAIT_FOR_SESSION}"
  run_one 0 gdkvm_echo gdkvm_echo echonet echonet_ed2es_endpoint "${DATASETS_ROOT}/processed/echonet_png128_10f" 8 4 "${PROJECT_DIR}/${LOG_DIR}/gdkvm_echo.log" \
    2>&1 | tee "${LOG_DIR}/gdkvm_echo.log"
}

run_camus_queue() {
  wait_for_session "${WAIT_FOR_SESSION}"
  run_one 1 gdkvm_camus gdkvm_camus camus camus_short_dense "${DATASETS_ROOT}/processed/camus_png256_10f" 4 4 "${PROJECT_DIR}/${LOG_DIR}/gdkvm_camus.log" \
    2>&1 | tee "${LOG_DIR}/gdkvm_camus.log"
}

tmux new-session -d -s "${SESSION_NAME}" -n echo "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START gdkvm echo baseline queue' && $(declare -f wait_for_session run_one run_echo_queue); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; LOG_DIR='${LOG_DIR}'; WAIT_FOR_SESSION='${WAIT_FOR_SESSION}'; run_echo_queue; echo '[\$(date +%F\\ %T)] END gdkvm echo baseline queue'; exec bash"
tmux new-window -t "${SESSION_NAME}" -n camus "cd '${PROJECT_DIR}' && echo '[\$(date +%F\\ %T)] START gdkvm camus baseline queue' && $(declare -f wait_for_session run_one run_camus_queue); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; LOG_DIR='${LOG_DIR}'; WAIT_FOR_SESSION='${WAIT_FOR_SESSION}'; run_camus_queue; echo '[\$(date +%F\\ %T)] END gdkvm camus baseline queue'; exec bash"

echo "Started tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Waiting for session: ${WAIT_FOR_SESSION:-<none>}"
echo "Logs:"
echo "  ${PROJECT_DIR}/${LOG_DIR}/gdkvm_echo.log"
echo "  ${PROJECT_DIR}/${LOG_DIR}/gdkvm_camus.log"
