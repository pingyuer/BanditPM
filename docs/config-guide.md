# Config Guide

Hydra config roots are under `configs/`.

```text
configs/model/gdkvm.yaml
configs/model/dpfr.yaml
configs/data/*.yaml
configs/runtime/default.yaml
configs/schedule/default_3k.yaml
configs/gdkvm_*.yaml
configs/dpfr_*.yaml
```

Experiment configs should compose model, data, runtime, schedule, then `_self_`.
New method configs should not edit `train.py`; they should register model/loss
providers and add `configs/model/<method>.yaml`.

No-leak defaults for comparable runs:

```yaml
phase_init:
  train: pred_or_zero
  val: pred_or_zero
  test: pred_or_zero
evaluation:
  init_mode: pred_or_zero
  exclude_init_frame: true
  protocol_version: v3_canonical_no_leak
model:
  allow_oracle_init_when_requested: false
  use_first_frame_gt_init: false
```
