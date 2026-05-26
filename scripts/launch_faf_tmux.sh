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

# Create two active training windows per GPU for the A30 pair.
tmux new-session -d -s "${SESSION_NAME}" -n "g0a"
tmux new-window -t "${SESSION_NAME}" -n "g0b"
tmux new-window -t "${SESSION_NAME}" -n "g1a"
tmux new-window -t "${SESSION_NAME}" -n "g1b"
tmux new-window -t "${SESSION_NAME}" -n "watch_echo"

printf '%s\n' "${SESSION_NAME}" > /tmp/faf_session
printf '%s\n' "${TIMESTAMP}" > /tmp/faf_ts

echo "============================================"
echo "FAF Experiment Suite: ${TIMESTAMP}"
echo "Session: ${SESSION_NAME}"
echo "4 training windows, 2 per GPU"
echo "============================================"

# --- Window g0a (GPU 0): CAMUS baseline ---
launch_run g0a 0 "camus_baseline" "faf_camus.yaml" \
    "exp_id=faf_camus_baseline" \
    "main_training.num_iterations=4000" \
    "main_training.batch_size=12" \
    "main_training.num_workers=10"

# --- Window g0b (GPU 0): CAMUS no-update ablation ---
launch_run g0b 0 "camus_no_update" "faf_camus.yaml" \
    "exp_id=faf_camus_no_update" \
    "main_training.num_iterations=4000" \
    "main_training.batch_size=12" \
    "main_training.num_workers=10" \
    "model.unext_faf.enable_memory_update=false"

# --- Window g1a (GPU 1): CAMUS single-anchor ablation ---
launch_run g1a 1 "camus_single_anchor" "faf_camus.yaml" \
    "exp_id=faf_camus_single_anchor" \
    "main_training.num_iterations=4000" \
    "main_training.batch_size=12" \
    "main_training.num_workers=10" \
    "model.unext_faf.num_anchors=1"

# --- Window g1b (GPU 1): CAMUS no proposal-to-residual ablation ---
launch_run g1b 1 "camus_no_proposal_residual" "faf_camus.yaml" \
    "exp_id=faf_camus_no_proposal_residual" \
    "main_training.num_iterations=4000" \
    "main_training.batch_size=12" \
    "main_training.num_workers=10" \
    "model.unext_faf.disable_proposal_in_residual=true"

tmux send-keys -t "${SESSION_NAME}:watch_echo" "cd '${PROJECT_DIR}' && bash scripts/watch_and_launch_echo.sh 2>&1 | tee '${LOG_DIR}/watch_echo.log'; exec bash" C-m

echo ""
echo "CAMUS experiments launched. Waiting for completion before EchoNet..."
echo "Monitor: tmux attach -t ${SESSION_NAME}"
echo "Logs:    ${LOG_DIR}/"
echo ""
echo "After CAMUS finishes, EchoNet experiments will be queued."
echo "============================================"
