#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tahara/GDKVM}"
UV_BIN="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
SESSION_NAME="${SESSION_NAME:-geomaskformer_3dataset_search_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/geomaskformer_3dataset_search_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/geomaskformer_3dataset_search_$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not available."
  exit 1
fi

COMMON_ARGS=(
  phase_init.train=pred_or_zero
  phase_init.val=pred_or_zero
  phase_init.test=pred_or_zero
  evaluation.init_mode=pred_or_zero
  evaluation.exclude_init_frame=true
  evaluation.protocol_version=v3_canonical_no_leak
  mlflow.stage=full
  mlflow.required=true
  mlflow.artifacts_enabled=true
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
  main_training.batch_size=4
  main_training.num_workers=4
)

run_one() {
  local gpu="$1"
  local config_name="$2"
  local exp_id="$3"
  local log_path="$4"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=. \
    HYDRA_FULL_ERROR=1 \
    DATASETS_ROOT="${DATASETS_ROOT}" \
    NO_PROXY="localhost,127.0.0.1,::1,172.16.240.77" \
    no_proxy="localhost,127.0.0.1,::1,172.16.240.77" \
    "${UV_BIN}" run python train.py \
    --config-name "${config_name}" \
    "${COMMON_ARGS[@]}" \
    "exp_id=${exp_id}" \
    "mlflow.command_log_path=${PROJECT_DIR}/${log_path}" \
    "hydra.run.dir=${OUT_DIR}/${exp_id}/\${now:%H-%M-%S}"
}

tmux new-session -d -s "${SESSION_NAME}" -n gpu0 \
  "cd '${PROJECT_DIR}' && $(declare -f run_one); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; OUT_DIR='${OUT_DIR}'; echo '[\$(date +%F\\ %T)] START adult fullres_cascade'; run_one 0 geomaskformer_fullres_cascade_echonet_adult geomaskformer_fullres_cascade_echonet_adult '${LOG_DIR}/adult_fullres_cascade.log' 2>&1 | tee '${LOG_DIR}/adult_fullres_cascade.log'; echo '[\$(date +%F\\ %T)] START camus cascade'; run_one 0 geomaskformer_cascade_refine_camus geomaskformer_cascade_refine_camus '${LOG_DIR}/camus_cascade_refine.log' 2>&1 | tee '${LOG_DIR}/camus_cascade_refine.log'; echo '[\$(date +%F\\ %T)] END gpu0 queue'; exec bash"

tmux new-window -t "${SESSION_NAME}" -n gpu1 \
  "cd '${PROJECT_DIR}' && $(declare -f run_one); COMMON_ARGS=($(printf '%q ' "${COMMON_ARGS[@]}")); PROJECT_DIR='${PROJECT_DIR}'; UV_BIN='${UV_BIN}'; DATASETS_ROOT='${DATASETS_ROOT}'; OUT_DIR='${OUT_DIR}'; echo '[\$(date +%F\\ %T)] START pediatric cascade'; run_one 1 geomaskformer_cascade_refine_echonet_pediatric geomaskformer_cascade_refine_echonet_pediatric '${LOG_DIR}/pediatric_cascade_refine.log' 2>&1 | tee '${LOG_DIR}/pediatric_cascade_refine.log'; echo '[\$(date +%F\\ %T)] START pediatric fullres_cascade'; run_one 1 geomaskformer_fullres_cascade_echonet_pediatric geomaskformer_fullres_cascade_echonet_pediatric '${LOG_DIR}/pediatric_fullres_cascade.log' 2>&1 | tee '${LOG_DIR}/pediatric_fullres_cascade.log'; echo '[\$(date +%F\\ %T)] START camus fullres_cascade'; run_one 1 geomaskformer_fullres_cascade_camus geomaskformer_fullres_cascade_camus '${LOG_DIR}/camus_fullres_cascade.log' 2>&1 | tee '${LOG_DIR}/camus_fullres_cascade.log'; echo '[\$(date +%F\\ %T)] END gpu1 queue'; exec bash"

echo "Started tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Logs:"
echo "  ${PROJECT_DIR}/${LOG_DIR}/adult_fullres_cascade.log"
echo "  ${PROJECT_DIR}/${LOG_DIR}/pediatric_cascade_refine.log"
echo "  ${PROJECT_DIR}/${LOG_DIR}/pediatric_fullres_cascade.log"
echo "  ${PROJECT_DIR}/${LOG_DIR}/camus_cascade_refine.log"
echo "  ${PROJECT_DIR}/${LOG_DIR}/camus_fullres_cascade.log"
