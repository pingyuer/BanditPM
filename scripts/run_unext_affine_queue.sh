#!/usr/bin/env bash
set -euo pipefail

role="${1:?role is required: camus_main|camus_ablation|echo_main|echo_ablation}"
run_root="${2:?run root is required}"
gpu="${3:?gpu index is required}"

uv_bin="${UV_BIN:-/home/tahara/miniconda3/bin/uv}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

common_overrides=(
  save=1
  save_weights_interval=500
  save_checkpoint_interval=500
  mlflow.required=false
  mlflow.artifacts_required=false
)

dataset_overrides() {
  local dataset="$1"
  if [[ "${dataset}" == "camus" ]]; then
    printf "%s\n" \
      main_training.batch_size=6 \
      main_training.num_workers=8
  else
    printf "%s\n" \
      main_training.batch_size=12 \
      main_training.num_workers=10
  fi
}

ckpt_for() {
  local stage0_dir="$1"
  if [[ -f "${stage0_dir}/best_raw.pth" ]]; then
    printf "%s\n" "${stage0_dir}/best_raw.pth"
  elif [[ -f "${stage0_dir}/latest_weights.pth" ]]; then
    printf "%s\n" "${stage0_dir}/latest_weights.pth"
  elif [[ -f "${stage0_dir}/latest.pth" ]]; then
    printf "%s\n" "${stage0_dir}/latest.pth"
  else
    return 1
  fi
}

wait_for_ckpt() {
  local stage0_dir="$1"
  local waited=0
  while ! ckpt_for "${stage0_dir}" >/dev/null 2>&1; do
    if (( waited >= 172800 )); then
      echo "Timed out waiting for checkpoint in ${stage0_dir}" >&2
      return 1
    fi
    echo "Waiting for Stage0 checkpoint in ${stage0_dir} ..."
    sleep 60
    waited=$((waited + 60))
  done
  ckpt_for "${stage0_dir}"
}

run_stage0() {
  local dataset="$1"
  local config="$2"
  local out_dir="$3"
  local seed="$4"
  echo "Starting Stage0 ${dataset} on GPU ${gpu}: ${out_dir}"
  "${uv_bin}" run python train.py \
    --config-name "${config}" \
    seed="${seed}" \
    exp_id="stage0_${dataset}_anchor_s${seed}" \
    hydra.run.dir="${out_dir}" \
    $(dataset_overrides "${dataset}") \
    "${common_overrides[@]}"
}

run_faf() {
  local dataset="$1"
  local config="$2"
  local out_dir="$3"
  local seed="$4"
  local mode="$5"
  local ckpt="$6"
  echo "Starting FAF ${dataset} mode=${mode} on GPU ${gpu}: ${out_dir}"
  echo "Using pretrained UNeXt checkpoint: ${ckpt}"
  "${uv_bin}" run python train.py \
    --config-name "${config}" \
    seed="${seed}" \
    exp_id="faf_${dataset}_${mode}_s${seed}" \
    model.unext_faf.pretrained_unext_path="${ckpt}" \
    model.unext_faf.prediction_mode="${mode}" \
    hydra.run.dir="${out_dir}" \
    $(dataset_overrides "${dataset}") \
    "${common_overrides[@]}"
}

case "${role}" in
  camus_main)
    stage0_dir="${run_root}/camus_stage0_s42"
    run_stage0 camus unext_anchor_warmup_camus.yaml "${stage0_dir}" 42
    ckpt="$(ckpt_for "${stage0_dir}")"
    run_faf camus faf_camus.yaml "${run_root}/camus_faf_affine_mixture_safe_s42" 42 affine_mixture_safe "${ckpt}"
    ;;
  camus_ablation)
    stage0_dir="${run_root}/camus_stage0_s42"
    ckpt="$(wait_for_ckpt "${stage0_dir}")"
    run_faf camus faf_camus.yaml "${run_root}/camus_faf_affine_no_temporal_s43" 43 affine_no_temporal "${ckpt}"
    ;;
  echo_main)
    stage0_dir="${run_root}/echo_stage0_s42"
    run_stage0 echo unext_anchor_warmup_echo.yaml "${stage0_dir}" 42
    ckpt="$(ckpt_for "${stage0_dir}")"
    run_faf echo faf_echo.yaml "${run_root}/echo_faf_affine_mixture_safe_s42" 42 affine_mixture_safe "${ckpt}"
    ;;
  echo_ablation)
    stage0_dir="${run_root}/echo_stage0_s42"
    ckpt="$(wait_for_ckpt "${stage0_dir}")"
    run_faf echo faf_echo.yaml "${run_root}/echo_faf_affine_no_temporal_s43" 43 affine_no_temporal "${ckpt}"
    ;;
  *)
    echo "Unknown role: ${role}" >&2
    exit 2
    ;;
esac
