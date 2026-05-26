#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/tahara/GDKVM"
UV_BIN="/home/tahara/miniconda3/bin/uv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_DIR}/logs/exp_${TIMESTAMP}"
SESSION_NAME="faf_exp_${TIMESTAMP}"

mkdir -p "${LOG_DIR}"

# Common env setup (will be embedded in each tmux command)
ENV_SETUP="cd ${PROJECT_DIR} && export PYTHONPATH=${PROJECT_DIR} HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

launch_run() {
    local win="$1"
    local gpu="$2"
    local name="$3"
    local config="$4"
    shift 4
    local overrides=("$@")

    local port=$((RANDOM % 40000 + 20000))
    local log_file="${LOG_DIR}/${name}.log"
    local out_dir="outputs/exp_${TIMESTAMP}/${name}"

    local override_str=""
    for o in "${overrides[@]}"; do
        override_str="${override_str} ${o}"
    done

    local cmd="${ENV_SETUP} && CUDA_VISIBLE_DEVICES=${gpu} MASTER_ADDR=127.0.0.1 MASTER_PORT=${port} ${UV_BIN} run torchrun --standalone --nproc_per_node=1 train.py --config-name ${config} ${override_str} hydra.run.dir=${out_dir}/\\\${now:%H-%M-%S} 2>&1 | tee ${log_file}"

    tmux send-keys -t "${SESSION_NAME}:${win}" "${cmd}" C-m
    echo "  Window ${win} [GPU ${gpu}]: ${name}"
}

# Create tmux session with 4 windows
tmux new-session -d -s "${SESSION_NAME}" -n "gpu0a"
tmux new-window -t "${SESSION_NAME}" -n "gpu0b"
tmux new-window -t "${SESSION_NAME}" -n "gpu1a"
tmux new-window -t "${SESSION_NAME}" -n "gpu1b"

echo "============================================"
echo "FAF Experiment Suite: ${TIMESTAMP}"
echo "Session: ${SESSION_NAME}"
echo "4 tmux windows, 2 per GPU"
echo "============================================"

# --- Window 0 (GPU 0): camus_baseline → echo_baseline ---
launch_run 0 0 "camus_baseline" "functional_anchor_camus.yaml" \
    "exp_id=faf_baseline_camus" \
    "main_training.num_iterations=4000" \
    "main_training.batch_size=24"

# --- Window 1 (GPU 0): camus_anchor_primary → echo_anchor_primary ---
launch_run 1 0 "camus_anchor_primary" "functional_anchor_camus.yaml" \
    "exp_id=faf_anchor_primary_camus" \
    "main_training.num_iterations=4000" \
    "main_training.batch_size=24" \
    "model.functional_anchor.prediction_mode=anchor_primary" \
    "model.functional_anchor.residual_scale.init=0.05" \
    "model.functional_anchor.residual_scale.max=0.25" \
    "model.functional_anchor.lambda_base_seg=0.2" \
    "model.functional_anchor.lambda_anchor=0.5"

# --- Window 2 (GPU 1): camus_more_anchors → echo_more_anchors ---
launch_run 2 1 "camus_more_anchors" "functional_anchor_camus.yaml" \
    "exp_id=faf_more_anchors_camus" \
    "main_training.num_iterations=4000" \
    "main_training.batch_size=24" \
    "model.functional_anchor.num_slots=8" \
    "model.functional_anchor.lambda_anchor=0.5" \
    "model.functional_anchor.lambda_base_seg=0.3" \
    "model.functional_anchor.lambda_slot_area_order=0.05" \
    "model.functional_anchor.lambda_phase_slot_correlation=0.02"

# --- Window 3 (GPU 1): camus_strong_mod → echo_strong_mod ---
launch_run 3 1 "camus_strong_mod" "functional_anchor_camus.yaml" \
    "exp_id=faf_strong_mod_camus" \
    "main_training.num_iterations=4000" \
    "main_training.batch_size=24" \
    "model.functional_anchor.residual_scale.init=0.05" \
    "model.functional_anchor.residual_scale.max=0.30" \
    "model.functional_anchor.residual_scale.warmup_iters=300" \
    "model.functional_anchor.lambda_base_seg=0.3" \
    "model.functional_anchor.lambda_residual_smallness=0.02" \
    "model.functional_anchor.lambda_boundary_residual=0.15"

echo ""
echo "CAMUS experiments launched. Waiting for completion before EchoNet..."
echo "Monitor: tmux attach -t ${SESSION_NAME}"
echo "Logs:    ${LOG_DIR}/"
echo ""
echo "After CAMUS finishes, EchoNet experiments will be queued."
echo "============================================"
