#!/usr/bin/env bash
set -euo pipefail

cd /home/tahara/GDKVM
mkdir -p logs outputs/proto_ablation/camus

run_one() {
  local config_name="$1"
  local exp_name="$2"
  shift 2

  ./.venv/bin/python train.py \
    --config-name="${config_name}" \
    hydra.run.dir="/home/tahara/GDKVM/outputs/proto_ablation/camus/${exp_name}" \
    main_training.batch_size=8 \
    main_training.num_workers=8 \
    main_training.num_iterations=1000 \
    'main_training.lr_schedule_steps=[333,667]' \
    eval_stage.eval_interval=200 \
    save=0 \
    wandb_mode=offline \
    exp_id="${exp_name}" \
    dataset_name=camus \
    data_path=/home/tahara/datasets/processed/camus_full_png256_10f \
    data.protocol_name=camus_full_dense \
    "$@" \
    2>&1 | tee "logs/${exp_name}.log"
}

run_one camus_full_dense_predinit.yaml camus_full_dense_original_gdr_predinit \
  phase_init.train=pred_or_zero phase_init.val=pred_or_zero phase_init.test=pred_or_zero \
  evaluation.init_mode=pred_or_zero evaluation.frame_scope=all_available

run_one camus_full_dense_predinit.yaml camus_full_dense_kpff_predinit \
  phase_init.train=pred_or_zero phase_init.val=pred_or_zero phase_init.test=pred_or_zero \
  evaluation.init_mode=pred_or_zero evaluation.frame_scope=all_available \
  model.memory_core.type=none model.temporal_memory.type=none

run_one camus_full_dense_predinit.yaml camus_full_dense_bpm_rule_predinit \
  phase_init.train=pred_or_zero phase_init.val=pred_or_zero phase_init.test=pred_or_zero \
  evaluation.init_mode=pred_or_zero evaluation.frame_scope=all_available \
  model.memory_core.type=bpm model.temporal_memory.type=bpm \
  model.temporal_memory.bpm.ENABLE=true \
  model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true \
  model.temporal_memory.bpm.USE_LEARNED_POLICY=false \
  model.temporal_memory.bpm.EXEC_POLICY=rule \
  model.temporal_memory.bpm.ENABLE_POLICY_LOSS=false \
  model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=false \
  model.temporal_memory.bpm.ENABLE_RL_LOSS=false

run_one camus_full_dense_predinit.yaml camus_full_dense_bpm_rl_predinit \
  phase_init.train=pred_or_zero phase_init.val=pred_or_zero phase_init.test=pred_or_zero \
  evaluation.init_mode=pred_or_zero evaluation.frame_scope=all_available \
  model.memory_core.type=bpm model.temporal_memory.type=bpm \
  model.temporal_memory.bpm.ENABLE=true \
  model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true \
  model.temporal_memory.bpm.USE_LEARNED_POLICY=true \
  model.temporal_memory.bpm.EXEC_POLICY=mixed \
  model.temporal_memory.bpm.ENABLE_POLICY_LOSS=true \
  model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=true \
  model.temporal_memory.bpm.ENABLE_RL_LOSS=true

run_one camus_full_dense_oracle.yaml camus_full_dense_original_gdr_oracle \
  phase_init.train=oracle_gt phase_init.val=oracle_gt phase_init.test=oracle_gt \
  evaluation.init_mode=oracle_gt evaluation.frame_scope=all_available

run_one camus_full_dense_oracle.yaml camus_full_dense_kpff_oracle \
  phase_init.train=oracle_gt phase_init.val=oracle_gt phase_init.test=oracle_gt \
  evaluation.init_mode=oracle_gt evaluation.frame_scope=all_available \
  model.memory_core.type=none model.temporal_memory.type=none

run_one camus_full_dense_oracle.yaml camus_full_dense_bpm_rule_oracle \
  phase_init.train=oracle_gt phase_init.val=oracle_gt phase_init.test=oracle_gt \
  evaluation.init_mode=oracle_gt evaluation.frame_scope=all_available \
  model.memory_core.type=bpm model.temporal_memory.type=bpm \
  model.temporal_memory.bpm.ENABLE=true \
  model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true \
  model.temporal_memory.bpm.USE_LEARNED_POLICY=false \
  model.temporal_memory.bpm.EXEC_POLICY=rule \
  model.temporal_memory.bpm.ENABLE_POLICY_LOSS=false \
  model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=false \
  model.temporal_memory.bpm.ENABLE_RL_LOSS=false

run_one camus_full_dense_oracle.yaml camus_full_dense_bpm_rl_oracle \
  phase_init.train=oracle_gt phase_init.val=oracle_gt phase_init.test=oracle_gt \
  evaluation.init_mode=oracle_gt evaluation.frame_scope=all_available \
  model.memory_core.type=bpm model.temporal_memory.type=bpm \
  model.temporal_memory.bpm.ENABLE=true \
  model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true \
  model.temporal_memory.bpm.USE_LEARNED_POLICY=true \
  model.temporal_memory.bpm.EXEC_POLICY=mixed \
  model.temporal_memory.bpm.ENABLE_POLICY_LOSS=true \
  model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=true \
  model.temporal_memory.bpm.ENABLE_RL_LOSS=true
