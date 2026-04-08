# BanditPM Experiment Guide

## Overview

This repository currently supports two EchoNet-style task protocols under the same training shell.

- `ED->ES 10f`
  - Short-range propagation between the two traced keyframes.
  - Sparse endpoint supervision.
  - Existing processed roots:
    - `/home/tahara/datasets/processed/echonet_png128_10f`
    - `/home/tahara/datasets/processed/echonet_pediatric_a4c_png128_10f`
- `Full-cycle 10f`
  - Long-range propagation over an estimated full cardiac cycle.
  - Sparse keyframe supervision.
  - Processed roots created by preprocessing:
    - `/home/tahara/datasets/processed/echonet_full_cycle_png128_10f`
    - `/home/tahara/datasets/processed/echonet_pediatric_a4c_full_cycle_png128_10f`

Both tasks keep the same training interface:

- `dataset_name=echonet`
- `10` grayscale frames per clip
- sparse mask supervision through `label_valid`
- first-frame prompt through `ff_gt`

## Methods

The current baseline matrix uses four methods.

- `gdkvm`
  - Original `KPFF + GDR`
- `kpff`
  - `KPFF` only, temporal memory disabled
- `bpm_rule`
  - `KPFF + BPM` with rule-based maintenance
- `bpm_rl`
  - `KPFF + BPM` with learned/RL maintenance

## Data Protocols

### ED->ES 10f

- Use the two traced frames as temporal endpoints.
- Uniformly sample `10` frames between them.
- Save labels only at `0000.png` and `0009.png`.

### Full-cycle 10f

- Let the two traced frames be `t0 < t1`.
- Estimate cycle end as `t_end = min(video_len - 1, t0 + 2 * (t1 - t0))`.
- Sample the clip in two segments:
  - `t0 -> t1` fills the first half of the clip
  - `t1 -> t_end` fills the second half
- This anchors the second traced frame at a fixed clip position.
- Save labels at:
  - `0000.png`
  - `0005.png`
- This keeps sparse supervision while increasing temporal span.

### Pediatric Split Mapping

For pediatric `A4C`, folds are mapped to train/val/test as:

- `train = 0,1,2,3,4,5,6,7`
- `val = 8`
- `test = 9`

## Preprocessing Commands

Generate `EchoNet-Dynamic Full-cycle 10f`:

```bash
./.venv/bin/python tools/preprocess_echonet.py \
  --input_root /home/tahara/datasets \
  --output_root /home/tahara/datasets/processed \
  --sampling_mode full_cycle \
  --overwrite
```

Generate `Pediatric A4C Full-cycle 10f`:

```bash
./.venv/bin/python tools/preprocess_echonet_pediatric.py \
  --input_root /home/tahara/datasets/echonetpediatric \
  --output_root /home/tahara/datasets/processed \
  --view A4C \
  --sampling_mode full_cycle \
  --overwrite
```

## Training Commands

Single-run examples:

### EchoNet-Dynamic Full-cycle 10f

```bash
/home/tahara/miniconda3/bin/uv run python train.py \
  --config-name config_banditpm_baseline \
  exp_id=echonet_full_cycle_gdkvm \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_full_cycle_png128_10f
```

### Pediatric A4C Full-cycle 10f

```bash
/home/tahara/miniconda3/bin/uv run python train.py \
  --config-name config_banditpm_baseline \
  exp_id=echonet_pediatric_a4c_full_cycle_gdkvm \
  wandb_mode=online \
  dataset_name=echonet \
  data_path=/home/tahara/datasets/processed/echonet_pediatric_a4c_full_cycle_png128_10f
```

## Tmux Queues

The full-cycle batch experiments are launched through:

- `/home/tahara/GDKVM/scripts/run_fullcycle_gpu0.sh`
- `/home/tahara/GDKVM/scripts/run_fullcycle_gpu1.sh`

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
tail -f outputs/BanditPM/tmux_logs/echonet_pediatric_a4c_full_cycle_gdkvm.log
```

## Current Full-cycle Queue Layout

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

All queue runs use:

- `eval_stage.eval_interval=200`
- `wandb_mode=online`
- `save_weights_interval=500`
