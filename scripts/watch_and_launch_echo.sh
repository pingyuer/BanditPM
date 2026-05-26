#!/usr/bin/env bash
set -euo pipefail

SESSION="$(cat /tmp/faf_session)"
TS="$(cat /tmp/faf_ts)"
LOG_DIR="/home/tahara/GDKVM/logs/exp_${TS}"
PROJ="/home/tahara/GDKVM"
UV="/home/tahara/miniconda3/bin/uv"

echo "[$(date '+%H:%M:%S')] Waiting for CAMUS experiments..."

has_training_descendant() {
    local root_pid="$1"
    local queue=("$root_pid")
    local current child cmd

    while [ "${#queue[@]}" -gt 0 ]; do
        current="${queue[0]}"
        queue=("${queue[@]:1}")

        while read -r child; do
            [ -z "$child" ] && continue
            cmd="$(ps -p "$child" -o args= 2>/dev/null || true)"
            if [[ "$cmd" == *"torchrun"* || "$cmd" == *"train.py"* ]]; then
                return 0
            fi
            queue+=("$child")
        done < <(pgrep -P "$current" 2>/dev/null || true)
    done

    return 1
}

while true; do
    alive=0
    for win in g0a g0b g1a g1b; do
        pane_pid=$(tmux display-message -t "${SESSION}:${win}" -p '#{pane_pid}' 2>/dev/null || echo "")
        if [ -n "$pane_pid" ] && has_training_descendant "$pane_pid"; then
            alive=$((alive + 1))
        fi
    done
    if [ "$alive" -eq 0 ]; then break; fi
    echo "[$(date '+%H:%M:%S')] ${alive} CAMUS jobs running..."
    sleep 60
done

echo "[$(date '+%H:%M:%S')] CAMUS done. Launching EchoNet..."
ENV="cd ${PROJ} && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && export PYTHONPATH=${PROJ} HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

tmux send-keys -t "${SESSION}:g0a" "${ENV} && CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=31201 ${UV} run torchrun --standalone --nproc_per_node=1 train.py --config-name faf_echo.yaml exp_id=faf_echo_baseline main_training.batch_size=12 main_training.num_workers=10 hydra.run.dir=outputs/exp_${TS}/echo_baseline/\${now:%H-%M-%S} 2>&1 | tee ${LOG_DIR}/echo_baseline.log" C-m

tmux send-keys -t "${SESSION}:g1a" "${ENV} && CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=31202 ${UV} run torchrun --standalone --nproc_per_node=1 train.py --config-name faf_echo.yaml exp_id=faf_echo_no_update model.unext_faf.enable_memory_update=false main_training.batch_size=12 main_training.num_workers=10 hydra.run.dir=outputs/exp_${TS}/echo_no_update/\${now:%H-%M-%S} 2>&1 | tee ${LOG_DIR}/echo_no_update.log" C-m

tmux send-keys -t "${SESSION}:g0b" "${ENV} && CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=31203 ${UV} run torchrun --standalone --nproc_per_node=1 train.py --config-name faf_echo.yaml exp_id=faf_echo_single_anchor model.unext_faf.num_anchors=1 main_training.batch_size=12 main_training.num_workers=10 hydra.run.dir=outputs/exp_${TS}/echo_single_anchor/\${now:%H-%M-%S} 2>&1 | tee ${LOG_DIR}/echo_single_anchor.log" C-m

tmux send-keys -t "${SESSION}:g1b" "${ENV} && CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=31204 ${UV} run torchrun --standalone --nproc_per_node=1 train.py --config-name faf_echo.yaml exp_id=faf_echo_no_proposal_residual model.unext_faf.disable_proposal_in_residual=true main_training.batch_size=12 main_training.num_workers=10 hydra.run.dir=outputs/exp_${TS}/echo_no_proposal_residual/\${now:%H-%M-%S} 2>&1 | tee ${LOG_DIR}/echo_no_proposal_residual.log" C-m

echo "[$(date '+%H:%M:%S')] EchoNet launched."
