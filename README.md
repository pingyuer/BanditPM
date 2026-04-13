# BanditPM

## Overview

BanditPM is a research codebase built on top of the original GDKVM implementation.

This repository keeps the original training shell and most of the project layout, but we have already done a minimal-invasive refactor around the temporal-memory path so that we can compare:

- `gdkvm`: original `KPFF + GDR`
- `kpff`: `KPFF` only, temporal memory disabled
- `bpm_rule`: `KPFF + Bandit Prototype Maintenance` with rule-based maintenance
- `bpm_rl`: `KPFF + Bandit Prototype Maintenance` with learned/RL maintenance

The current goal is not to turn this repo into a generic plugin framework. The goal is to keep the GDKVM shell, preserve KPFF, and make the memory/update path easier to replace for BanditPM research.

## What We Changed

The main changes from the original baseline are:

- project naming is unified under `BanditPM`
- wandb defaults, output directory naming, and checkpoint naming now follow the `BanditPM` convention
- the temporal-memory path is collected into replaceable modules under `model/modules/`
- prototype-related components are kept in-place and routed through the same memory path
- EchoNet-style sparse supervision now supports both the original short-range task and a harder full-cycle task

This means we can now run the same training shell on:

- original short-range `ED->ES 10f`
- longer-range `Full-cycle 10f`
- original EchoNet-Dynamic
- EchoNet Pediatric A4C

## Current Task Protocols

We currently use two EchoNet-style task protocols.

### 1. ED->ES 10f

- use the two traced frames as temporal endpoints
- sample `10` grayscale frames between them
- save sparse labels at:
  - `0000.png`
  - `0009.png`

This is the short-range propagation baseline.

### 2. Full-cycle 10f

- let the traced frames be `t0 < t1`
- estimate cycle end as:
  - `t_end = min(video_len - 1, t0 + 2 * (t1 - t0))`
- sample the clip in two segments:
  - `t0 -> t1` fills the first half
  - `t1 -> t_end` fills the second half
- save sparse labels at fixed positions:
  - `0000.png`
  - `0005.png`

This keeps sparse supervision but makes the temporal span much harder than the original short-range setup.

## Datasets

### EchoNet-Dynamic

Processed root for the original short-range task:

- `/home/tahara/datasets/processed/echonet_png128_10f`

Processed root for the full-cycle task:

- `/home/tahara/datasets/processed/echonet_full_cycle_png128_10f`

Use it with:

```bash
./.venv/bin/python train.py \
  --config-name echonet_ed2es_endpoint_oracle
```

Full-cycle:

```bash
./.venv/bin/python train.py \
  --config-name echonet_fullcycle_oracle
```

### EchoNet Pediatric A4C

Processed root for the original short-range task:

- `/home/tahara/datasets/processed/echonet_pediatric_a4c_png128_10f`

Processed root for the full-cycle task:

- `/home/tahara/datasets/processed/echonet_pediatric_a4c_full_cycle_png128_10f`

Pediatric split mapping:

- `train = 0,1,2,3,4,5,6,7`
- `val = 8`
- `test = 9`

Use it with:

```bash
./.venv/bin/python train.py \
  --config-name config_banditpm_baseline \
  exp_id=echonet_pediatric_a4c_ed2es_gdkvm \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_pediatric_a4c_png128_10f \
  data.protocol_name=echonet_pediatric_a4c_endpoint
```

Full-cycle:

```bash
./.venv/bin/python train.py \
  --config-name config_banditpm_baseline \
  exp_id=echonet_pediatric_a4c_full_cycle_gdkvm \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_pediatric_a4c_full_cycle_png128_10f \
  data.protocol_name=echonet_pediatric_fullcycle_sparse
```

### CAMUS

Processed root for the short dense task:

- `/home/tahara/datasets/processed/camus_png256_10f`

Processed root for the full dense task:

- `/home/tahara/datasets/processed/camus_full_png256_10f`

Use it with:

```bash
./.venv/bin/python train.py \
  --config-name camus_short_dense_oracle
```

Full dense:

```bash
./.venv/bin/python train.py \
  --config-name camus_full_dense_oracle
```

At the moment, CAMUS is mainly kept as a supporting experiment rather than the main BanditPM stress test.

### CardiacUDA

Processed root for the current GDKVM integration:

- `/home/tahara/datasets/processed/cardiacuda_a4c_lv_png128_10f`

Current contract:

- view: `A4C`
- target: `LV` only
- supervision: sparse labels preserved from the annotated CardiacUDA frames
- default split mapping:
  `train = Site_G_100 + Site_R_126`
  `val = Site_G_20 + Site_R_52`
  `test = Site_G_29`
- skipped by default:
  `Site_R_73` because labels are absent in the released package
  `label_all_frame` because some cases use a different label encoding than the stable `Site_*` folders

Real local preprocessing result on `2026-04-12`:

- processed samples: `283`
- skipped samples: `1`
- split counts: `train=181`, `val=73`, `test=29`

Use it with:

```bash
./.venv/bin/python train.py \
  --config-name cardiacuda_a4c_lv_oracle
```

Pred-init:

```bash
./.venv/bin/python train.py \
  --config-name cardiacuda_a4c_lv_predinit
```

## Repository Entry Points

The main files you will usually touch are:

- [train.py](/home/tahara/GDKVM/train.py): training entry
- [train.sh](/home/tahara/GDKVM/train.sh): multi-GPU launch helper
- [model/gdkvm01.py](/home/tahara/GDKVM/model/gdkvm01.py): main model entry
- [model/kpff.py](/home/tahara/GDKVM/model/kpff.py): KPFF fusion module
- [model/trainer.py](/home/tahara/GDKVM/model/trainer.py): trainer and logging
- [model/modules/memory_core.py](/home/tahara/GDKVM/model/modules/memory_core.py): memory routing path
- [model/modules/gdr_core.py](/home/tahara/GDKVM/model/modules/gdr_core.py): original GDR path
- [model/modules/prototype_manager.py](/home/tahara/GDKVM/model/modules/prototype_manager.py): BPM path

## Environment

This repo is currently run from the local project virtual environment:

- `./.venv/bin/python`
- `./.venv/bin/torchrun`

For tests, using `uv` is fine, but make sure `PYTHONPATH=.`

Example:

```bash
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run pytest -q
```

## Preprocessing

### EchoNet-Dynamic ED->ES 10f

```bash
./.venv/bin/python tools/preprocess_echonet.py \
  --input_root /home/tahara/datasets \
  --output_root /home/tahara/datasets/processed \
  --sampling_mode ed_to_es \
  --overwrite
```

### EchoNet-Dynamic Full-cycle 10f

```bash
./.venv/bin/python tools/preprocess_echonet.py \
  --input_root /home/tahara/datasets \
  --output_root /home/tahara/datasets/processed \
  --sampling_mode full_cycle \
  --overwrite
```

### EchoNet Pediatric A4C ED->ES 10f

```bash
./.venv/bin/python tools/preprocess_echonet_pediatric.py \
  --input_root /home/tahara/datasets/echonetpediatric \
  --output_root /home/tahara/datasets/processed \
  --view A4C \
  --sampling_mode ed_to_es \
  --overwrite
```

### EchoNet Pediatric A4C Full-cycle 10f

```bash
./.venv/bin/python tools/preprocess_echonet_pediatric.py \
  --input_root /home/tahara/datasets/echonetpediatric \
  --output_root /home/tahara/datasets/processed \
  --view A4C \
  --sampling_mode full_cycle \
  --overwrite
```

### CAMUS Short Dense 10f

```bash
./.venv/bin/python tools/preprocess_camus.py \
  --input_root /home/tahara/datasets \
  --output_root /home/tahara/datasets/processed \
  --sampling_mode short \
  --overwrite
```

### CAMUS Full Dense 10f

```bash
./.venv/bin/python tools/preprocess_camus.py \
  --input_root /home/tahara/datasets \
  --output_root /home/tahara/datasets/processed \
  --sampling_mode full \
  --overwrite
```

### CardiacUDA A4C LV 10f

```bash
./.venv/bin/python tools/preprocess_cardiacuda.py \
  --input_root /home/tahara/datasets \
  --output_root /home/tahara/datasets/processed \
  --target_label 1 \
  --overwrite
```

Optional label mapping for CardiacUDA A4C:

- `1=LV`
- `2=LA`
- `3=RA`
- `4=RV`

The current checked-in config files target `A4C + LV`.

## How To Run

### Single run with `train.py`

EchoNet full-cycle `gdkvm`:

```bash
./.venv/bin/python train.py \
  --config-name config_banditpm_baseline \
  exp_id=echonet_full_cycle_gdkvm \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_full_cycle_png128_10f
```

Pediatric full-cycle `gdkvm`:

```bash
./.venv/bin/python train.py \
  --config-name config_banditpm_baseline \
  exp_id=echonet_pediatric_a4c_full_cycle_gdkvm \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_pediatric_a4c_full_cycle_png128_10f
```

### Multi-GPU launch with `train.sh`

```bash
bash train.sh --config-name config_banditpm_baseline
```

`train.sh` will:

- detect visible GPUs from `CUDA_VISIBLE_DEVICES`
- prefer `./.venv/bin/torchrun`
- set `PYTHONPATH`
- start distributed training with Hydra output directories

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash train.sh \
  --config-name config_banditpm_baseline \
  exp_id=echonet_full_cycle_gdkvm \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_full_cycle_png128_10f
```

## Batch Experiments With tmux

The current full-cycle queues are:

- [scripts/run_fullcycle_gpu0.sh](/home/tahara/GDKVM/scripts/run_fullcycle_gpu0.sh)
- [scripts/run_fullcycle_gpu1.sh](/home/tahara/GDKVM/scripts/run_fullcycle_gpu1.sh)

Launch:

```bash
tmux new-session -d -s fc_gpu0 'bash scripts/run_fullcycle_gpu0.sh'
tmux new-session -d -s fc_gpu1 'bash scripts/run_fullcycle_gpu1.sh'
```

Attach:

```bash
tmux attach -t fc_gpu0
tmux attach -t fc_gpu1
```

Logs:

```bash
tail -f outputs/BanditPM/tmux_logs/echonet_full_cycle_gdkvm.log
tail -f outputs/BanditPM/tmux_logs/echonet_full_cycle_kpff.log
```

Current queue layout:

GPU 0:

- `echonet_full_cycle_gdkvm`
- `echonet_full_cycle_bpm_rule`
- `echonet_pediatric_a4c_full_cycle_kpff`
- `echonet_pediatric_a4c_full_cycle_bpm_rl`

GPU 1:

- `echonet_full_cycle_kpff`
- `echonet_full_cycle_bpm_rl`
- `echonet_pediatric_a4c_full_cycle_gdkvm`
- `echonet_pediatric_a4c_full_cycle_bpm_rule`

Shared queue settings:

- `eval_stage.eval_interval=200`
- `wandb_mode=online`
- `save_weights_interval=500`

## wandb Naming

Default naming is now under the `BanditPM` project.

- `project`: `BanditPM`
- `group`: set by config or queue script
- `run name`: auto-generated from model name, dataset, experiment name, and seed
- `output dir`: `outputs/BanditPM/...`

## Tests

The current regression tests that matter for this workflow are:

- `tests/test_echonet_sampling.py`
- `tests/test_echo_dataset_sparse_labels.py`
- `tests/test_cardiacuda_preprocess.py`
- `tests/test_supervision_indices.py`
- `tests/test_gdkvm_temporal_modes.py`
- `tests/test_prototype_manager.py`

Run them with:

```bash
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run pytest -q \
  tests/test_echonet_sampling.py \
  tests/test_echo_dataset_sparse_labels.py \
  tests/test_cardiacuda_preprocess.py \
  tests/test_supervision_indices.py \
  tests/test_gdkvm_temporal_modes.py \
  tests/test_prototype_manager.py
```

## Reference

For the original published baseline, refer to the original GDKVM reproduction page:

- https://wangrui2025.github.io/GDKVM/reprod/index.html

## Citation

If this work helps your research, please cite the original GDKVM paper:

```bibtex
@InProceedings{Wang_ICCV25_GDKVM,
    author    = {Wang, Rui and Sun, Yimu and Guo, Jingxing and Wu, Huisi and Qin, Jing},
    title     = {{GDKVM}: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {12191-12200}
}
```

## Acknowledgement

This repository is built upon:

- [Cutie](https://github.com/hkchengrex/Cutie)
- [LiVOS](https://github.com/uncbiag/LiVOS)

We appreciate the original authors for maintaining strong open-source foundations.
