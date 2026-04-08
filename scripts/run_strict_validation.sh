#!/usr/bin/env bash
set -euo pipefail

cd /home/tahara/GDKVM
mkdir -p logs
export PYTHONPATH=.

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

run_and_log() {
  local log_path="$1"
  shift
  "$@" 2>&1 | tee "${log_path}"
}

run_and_log logs/pytest.log "${PYTHON_BIN}" -m pytest -q

run_and_log logs/smoke_camus.log \
  "${PYTHON_BIN}" train.py \
  --config-name=camus_short_dense_oracle.yaml \
  hydra.run.dir=outputs/smoke/camus_short_dense_oracle \
  main_training.batch_size=1 \
  main_training.num_iterations=1 \
  eval_stage.eval_interval=1 \
  save=0 \
  wandb_mode=offline

run_and_log logs/smoke_echonet_endpoint.log \
  "${PYTHON_BIN}" train.py \
  --config-name=echonet_ed2es_endpoint_predinit.yaml \
  hydra.run.dir=outputs/smoke/echonet_ed2es_endpoint_predinit \
  main_training.batch_size=1 \
  main_training.num_iterations=1 \
  eval_stage.eval_interval=1 \
  save=0 \
  wandb_mode=offline

run_and_log logs/smoke_echonet_fullcycle.log \
  "${PYTHON_BIN}" train.py \
  --config-name=echonet_fullcycle_predinit.yaml \
  hydra.run.dir=outputs/smoke/echonet_fullcycle_predinit \
  main_training.batch_size=1 \
  main_training.num_iterations=1 \
  eval_stage.eval_interval=1 \
  save=0 \
  wandb_mode=offline

run_and_log logs/full_validation.log \
  "${PYTHON_BIN}" scripts/run_all_proto_ablation.py
