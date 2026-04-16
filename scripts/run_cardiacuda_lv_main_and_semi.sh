#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
TRAIN_PY="$ROOT_DIR/train.py"
OUTPUT_ROOT="$ROOT_DIR/outputs/cardiacuda_protocols"
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
  local batch_size="$3"
  local num_workers="$4"
  shift 4

  local run_dir="$OUTPUT_ROOT/$exp_name"
  local log_file="$LOG_DIR/${exp_name}.log"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $exp_name"
  "$PYTHON_BIN" "$TRAIN_PY" \
    "--config-name=$config_name" \
    "hydra.run.dir=$run_dir" \
    "exp_id=$exp_name" \
    "main_training.batch_size=$batch_size" \
    "main_training.num_workers=$num_workers" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "$log_file"
}

run_method_suite() {
  local config_name="$1"
  local prefix="$2"
  local batch_size="$3"
  local num_workers="$4"

  run_one "$config_name" "${prefix}_original_gdr" "$batch_size" "$num_workers"

  run_one "$config_name" "${prefix}_kpff" "$batch_size" "$num_workers" \
    "model.memory_core.type=none" \
    "model.temporal_memory.type=none"

  run_one "$config_name" "${prefix}_bpm_rule" "$batch_size" "$num_workers" \
    "model.memory_core.type=bpm" \
    "model.temporal_memory.type=bpm" \
    "model.temporal_memory.bpm.ENABLE=true" \
    "model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true" \
    "model.temporal_memory.bpm.USE_LEARNED_POLICY=false" \
    "model.temporal_memory.bpm.EXEC_POLICY=rule" \
    "model.temporal_memory.bpm.ENABLE_POLICY_LOSS=false" \
    "model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=false" \
    "model.temporal_memory.bpm.ENABLE_RL_LOSS=false"

  run_one "$config_name" "${prefix}_bpm_rl" "$batch_size" "$num_workers" \
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

# Mainline: dense, fully supervised.
# The dense split is tiny (5/2/2 after filtering absent-LV cases), so keep the batch small
# to preserve multiple optimizer steps per effective epoch.
run_method_suite "cardiacuda_a4c_lv_dense_predinit.yaml" "cardiacuda_a4c_lv_dense_predinit" 2 4
run_method_suite "cardiacuda_a4c_lv_dense_oracle.yaml" "cardiacuda_a4c_lv_dense_oracle" 2 4

# Side branch: sparse, semi-supervised multi-anchor propagation.
# This split is much larger, so use a higher batch size to better utilize the A30.
run_method_suite "cardiacuda_a4c_lv_predinit.yaml" "cardiacuda_a4c_lv_sparse_predinit" 12 8
run_method_suite "cardiacuda_a4c_lv_oracle.yaml" "cardiacuda_a4c_lv_sparse_oracle" 12 8
