#!/usr/bin/env bash
set -euo pipefail

SESSION="$(cat /tmp/faf_session_name)"
TS="$(cat /tmp/faf_timestamp)"
LOG_DIR="/home/tahara/GDKVM/logs/exp_${TS}"
PROJ="/home/tahara/GDKVM"
UV="/home/tahara/miniconda3/bin/uv"

echo "Waiting for CAMUS experiments to finish..."
echo "Session: ${SESSION}"
echo "Log dir: ${LOG_DIR}"

# Wait for all 4 CAMUS processes to finish
while true; do
    alive=0
    for win in gpu0a gpu0b gpu1a gpu1b; do
        pane_pid=$(tmux display-message -t "${SESSION}:${win}" -p '#{pane_pid}' 2>/dev/null || echo "")
        if [ -n "$pane_pid" ]; then
            # Check if the pane's shell has child processes running
            children=$(pgrep -P "$pane_pid" 2>/dev/null | wc -l)
            if [ "$children" -gt 0 ]; then
                alive=$((alive + 1))
            fi
        fi
    done
    if [ "$alive" -eq 0 ]; then
        break
    fi
    echo "[$(date '+%H:%M:%S')] ${alive} CAMUS experiments still running..."
    sleep 60
done

echo "[$(date '+%H:%M:%S')] All CAMUS experiments finished. Launching EchoNet..."

ENV="cd ${PROJ} && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && export PYTHONPATH=${PROJ} HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# EchoNet baseline (GPU 0)
tmux send-keys -t "${SESSION}:gpu0a" "${ENV} && CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29601 ${UV} run torchrun --standalone --nproc_per_node=1 train.py --config-name functional_anchor_echo.yaml exp_id=faf_baseline_echo main_training.num_iterations=4000 main_training.batch_size=16 hydra.run.dir=outputs/exp_${TS}/echo_baseline/\${now:%H-%M-%S} 2>&1 | tee ${LOG_DIR}/echo_baseline.log" C-m

# EchoNet anchor_primary (GPU 0)
tmux send-keys -t "${SESSION}:gpu0b" "${ENV} && CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29602 ${UV} run torchrun --standalone --nproc_per_node=1 train.py --config-name functional_anchor_echo.yaml exp_id=faf_anchor_primary_echo main_training.num_iterations=4000 main_training.batch_size=16 model.functional_anchor.prediction_mode=anchor_primary model.functional_anchor.residual_scale.init=0.05 model.functional_anchor.residual_scale.max=0.25 model.functional_anchor.lambda_base_seg=0.2 model.functional_anchor.lambda_anchor=0.5 hydra.run.dir=outputs/exp_${TS}/echo_anchor_primary/\${now:%H-%M-%S} 2>&1 | tee ${LOG_DIR}/echo_anchor_primary.log" C-m

# EchoNet more_anchors (GPU 1)
tmux send-keys -t "${SESSION}:gpu1a" "${ENV} && CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29603 ${UV} run torchrun --standalone --nproc_per_node=1 train.py --config-name functional_anchor_echo.yaml exp_id=faf_more_anchors_echo main_training.num_iterations=4000 main_training.batch_size=16 model.functional_anchor.num_slots=8 model.functional_anchor.lambda_anchor=0.5 model.functional_anchor.lambda_base_seg=0.3 model.functional_anchor.lambda_slot_area_order=0.05 model.functional_anchor.lambda_phase_slot_correlation=0.02 hydra.run.dir=outputs/exp_${TS}/echo_more_anchors/\${now:%H-%M-%S} 2>&1 | tee ${LOG_DIR}/echo_more_anchors.log" C-m

# EchoNet strong_mod (GPU 1)
tmux send-keys -t "${SESSION}:gpu1b" "${ENV} && CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29604 ${UV} run torchrun --standalone --nproc_per_node=1 train.py --config-name functional_anchor_echo.yaml exp_id=faf_strong_mod_echo main_training.num_iterations=4000 main_training.batch_size=16 model.functional_anchor.residual_scale.init=0.05 model.functional_anchor.residual_scale.max=0.30 model.functional_anchor.residual_scale.warmup_iters=300 model.functional_anchor.lambda_base_seg=0.3 model.functional_anchor.lambda_residual_smallness=0.02 model.functional_anchor.lambda_boundary_residual=0.15 hydra.run.dir=outputs/exp_${TS}/echo_strong_mod/\${now:%H-%M-%S} 2>&1 | tee ${LOG_DIR}/echo_strong_mod.log" C-m

echo "[$(date '+%H:%M:%S')] EchoNet experiments launched."
echo "Monitor: tmux attach -t ${SESSION}"
