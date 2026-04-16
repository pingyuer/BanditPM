#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
TRAIN_PY="$ROOT_DIR/train.py"
OUTPUT_ROOT="$ROOT_DIR/outputs/cardiacuda_sparse_only"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

COMMON_ARGS=(
  "main_training.num_iterations=1000"
  "main_training.lr_schedule_steps=[333,667]"
  "eval_stage.eval_interval=200"
  "save=0"
  "wandb_mode=offline"
)

run_one() {
  local config_name="$1"
  local exp_name="$2"
  shift 2

  local run_dir="$OUTPUT_ROOT/$exp_name"
  local log_file="$LOG_DIR/${exp_name}.log"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $exp_name"
  "$PYTHON_BIN" "$TRAIN_PY" \
    "--config-name=$config_name" \
    "hydra.run.dir=$run_dir" \
    "exp_id=$exp_name" \
    "main_training.batch_size=12" \
    "main_training.num_workers=8" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "$log_file"
}

run_sparse_suite() {
  local config_name="$1"
  local prefix="$2"

  run_one "$config_name" "${prefix}_original_gdr"

  run_one "$config_name" "${prefix}_kpff" \
    "model.memory_core.type=none" \
    "model.temporal_memory.type=none"

  run_one "$config_name" "${prefix}_bpm_rule" \
    "model.memory_core.type=bpm" \
    "model.temporal_memory.type=bpm" \
    "model.temporal_memory.bpm.ENABLE=true" \
    "model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true" \
    "model.temporal_memory.bpm.USE_LEARNED_POLICY=false" \
    "model.temporal_memory.bpm.EXEC_POLICY=rule" \
    "model.temporal_memory.bpm.ENABLE_POLICY_LOSS=false" \
    "model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=false" \
    "model.temporal_memory.bpm.ENABLE_RL_LOSS=false"

  run_one "$config_name" "${prefix}_bpm_rl" \
    "model.memory_core.type=bpm" \
    "model.temporal_memory.type=bpm" \
    "model.temporal_memory.bpm.ENABLE=true" \
    "model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true" \
    "model.temporal_memory.bpm.USE_LEARNED_POLICY=true" \
    "model.temporal_memory.bpm.EXEC_POLICY=mixed" \
    "model.temporal_memory.bpm.ENABLE_POLICY_LOSS=true" \
    "model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=true" \
    "model.temporal_memory.bpm.ENABLE_RL_LOSS=true"
}

run_sparse_suite "cardiacuda_a4c_lv_predinit.yaml" "cardiacuda_a4c_lv_sparse_predinit"
run_sparse_suite "cardiacuda_a4c_lv_oracle.yaml" "cardiacuda_a4c_lv_sparse_oracle"
