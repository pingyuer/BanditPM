#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/tahara/GDKVM"
UV_BIN="/home/tahara/miniconda3/bin/uv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_DIR}/logs/exp_${TIMESTAMP}"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

run_experiment() {
    local gpu_id="$1"
    local name="$2"
    local config="$3"
    shift 3
    local overrides=("$@")

    local port=$((RANDOM % 40000 + 20000))
    local log_file="${LOG_DIR}/${name}.log"

    echo "[$(date '+%H:%M:%S')] Starting ${name} on GPU ${gpu_id} (port ${port})"
    echo "  Config: ${config}"
    echo "  Overrides: ${overrides[*]}"
    echo "  Log: ${log_file}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${port}" \
    "${UV_BIN}" run torchrun \
        --standalone \
        --nproc_per_node=1 \
        train.py \
        --config-name "${config}" \
        "${overrides[@]}" \
        hydra.run.dir="outputs/exp_${TIMESTAMP}/${name}/\${now:%H-%M-%S}" \
        2>&1 | tee "${log_file}"

    echo "[$(date '+%H:%M:%S')] Finished ${name}"
}

# ============================================================
# Experiment definitions
# ============================================================
# Variant 1: Baseline
# Variant 2: Anchor-primary (anchor drives prediction)
# Variant 3: More anchors (8 slots, stronger anchor loss)
# Variant 4: Strong modulation (larger residual scale)

CAMUS_BASELINE=(
    exp_id="faf_baseline_camus"
    main_training.num_iterations=4000
    main_training.batch_size=24
)

CAMUS_ANCHOR_PRIMARY=(
    exp_id="faf_anchor_primary_camus"
    main_training.num_iterations=4000
    main_training.batch_size=24
    model.functional_anchor.prediction_mode="anchor_primary"
    model.functional_anchor.residual_scale.init=0.05
    model.functional_anchor.residual_scale.max=0.25
    model.functional_anchor.lambda_base_seg=0.2
    model.functional_anchor.lambda_anchor=0.5
)

CAMUS_MORE_ANCHORS=(
    exp_id="faf_more_anchors_camus"
    main_training.num_iterations=4000
    main_training.batch_size=24
    model.functional_anchor.num_slots=8
    model.functional_anchor.lambda_anchor=0.5
    model.functional_anchor.lambda_base_seg=0.3
    model.functional_anchor.lambda_slot_area_order=0.05
    model.functional_anchor.lambda_phase_slot_correlation=0.02
)

CAMUS_STRONG_MOD=(
    exp_id="faf_strong_mod_camus"
    main_training.num_iterations=4000
    main_training.batch_size=24
    model.functional_anchor.residual_scale.init=0.05
    model.functional_anchor.residual_scale.max=0.30
    model.functional_anchor.residual_scale.warmup_iters=300
    model.functional_anchor.lambda_base_seg=0.3
    model.functional_anchor.lambda_residual_smallness=0.02
    model.functional_anchor.lambda_boundary_residual=0.15
)

ECHO_BASELINE=(
    exp_id="faf_baseline_echo"
    main_training.num_iterations=4000
    main_training.batch_size=24
)

ECHO_ANCHOR_PRIMARY=(
    exp_id="faf_anchor_primary_echo"
    main_training.num_iterations=4000
    main_training.batch_size=24
    model.functional_anchor.prediction_mode="anchor_primary"
    model.functional_anchor.residual_scale.init=0.05
    model.functional_anchor.residual_scale.max=0.25
    model.functional_anchor.lambda_base_seg=0.2
    model.functional_anchor.lambda_anchor=0.5
)

ECHO_MORE_ANCHORS=(
    exp_id="faf_more_anchors_echo"
    main_training.num_iterations=4000
    main_training.batch_size=24
    model.functional_anchor.num_slots=8
    model.functional_anchor.lambda_anchor=0.5
    model.functional_anchor.lambda_base_seg=0.3
    model.functional_anchor.lambda_slot_area_order=0.05
    model.functional_anchor.lambda_phase_slot_correlation=0.02
)

ECHO_STRONG_MOD=(
    exp_id="faf_strong_mod_echo"
    main_training.num_iterations=4000
    main_training.batch_size=24
    model.functional_anchor.residual_scale.init=0.05
    model.functional_anchor.residual_scale.max=0.30
    model.functional_anchor.residual_scale.warmup_iters=300
    model.functional_anchor.lambda_base_seg=0.3
    model.functional_anchor.lambda_residual_smallness=0.02
    model.functional_anchor.lambda_boundary_residual=0.15
)

# ============================================================
# Launch: 4 parallel jobs (2 per GPU)
# ============================================================
echo "============================================"
echo "FAF Experiment Suite: ${TIMESTAMP}"
echo "4 jobs parallel, 2 per GPU"
echo "============================================"

# --- Round 1: camus_baseline(GPU0) + echo_anchor_primary(GPU0) + camus_more_anchors(GPU1) + echo_strong_mod(GPU1) ---
run_experiment 0 "camus_baseline"      "functional_anchor_camus.yaml" "${CAMUS_BASELINE[@]}" &
PID1=$!
run_experiment 0 "echo_anchor_primary" "functional_anchor_echo.yaml"  "${ECHO_ANCHOR_PRIMARY[@]}" &
PID2=$!
run_experiment 1 "camus_more_anchors"  "functional_anchor_camus.yaml" "${CAMUS_MORE_ANCHORS[@]}" &
PID3=$!
run_experiment 1 "echo_strong_mod"     "functional_anchor_echo.yaml"  "${ECHO_STRONG_MOD[@]}" &
PID4=$!

echo "Round 1 PIDs: ${PID1} ${PID2} ${PID3} ${PID4}"
wait ${PID1} ${PID2} ${PID3} ${PID4}
echo "Round 1 complete."

# --- Round 2: camus_anchor_primary(GPU0) + echo_baseline(GPU0) + camus_strong_mod(GPU1) + echo_more_anchors(GPU1) ---
run_experiment 0 "camus_anchor_primary" "functional_anchor_camus.yaml" "${CAMUS_ANCHOR_PRIMARY[@]}" &
PID5=$!
run_experiment 0 "echo_baseline"       "functional_anchor_echo.yaml"  "${ECHO_BASELINE[@]}" &
PID6=$!
run_experiment 1 "camus_strong_mod"    "functional_anchor_camus.yaml" "${CAMUS_STRONG_MOD[@]}" &
PID7=$!
run_experiment 1 "echo_more_anchors"   "functional_anchor_echo.yaml"  "${ECHO_MORE_ANCHORS[@]}" &
PID8=$!

echo "Round 2 PIDs: ${PID5} ${PID6} ${PID7} ${PID8}"
wait ${PID5} ${PID6} ${PID7} ${PID8}
echo "Round 2 complete."

echo "============================================"
echo "All 8 experiments complete."
echo "Logs: ${LOG_DIR}"
echo "Outputs: outputs/exp_${TIMESTAMP}/"
echo "============================================"
