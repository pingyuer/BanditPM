# GDKVM / DPFR Research Workspace

This repository is an echocardiography video segmentation project. The public
project surface now exposes two runnable method families:

- `gdkvm`: the GDKVM path with its KPFF/GDR internals kept private to the method.
- `dpfr`: Dual-Prompt Flow Refinement with prompt, flow, and fusion diagnostics.

The codebase is being migrated to a standard `src/gdkvm_project/` layout. Legacy
top-level modules may still exist as implementation compatibility, but new code
should import from `gdkvm_project`.

## Entry Points

```bash
PYTHONPATH=src:. /home/tahara/miniconda3/bin/uv run python train.py --config-name gdkvm_echo
PYTHONPATH=src:. /home/tahara/miniconda3/bin/uv run python train.py --config-name dpfr_echo
```

Console scripts are also declared:

```bash
gdkvm-train --config-name dpfr_echo
gdkvm-visualize /path/to/run --split val
gdkvm-compare /path/to/local_run /path/to/other_server_run
```

## Configuration

Hydra configs live in `configs/`:

```text
configs/model/{gdkvm,dpfr}.yaml
configs/data/{echo,camus,domain,...}.yaml
configs/runtime/default.yaml
configs/schedule/default_3k.yaml
configs/gdkvm_*.yaml
configs/dpfr_*.yaml
```

Current canonical configs are the `gdkvm_*` and `dpfr_*` experiment files. All
formal no-leak runs should keep:

```yaml
evaluation:
  init_mode: pred_or_zero
  exclude_init_frame: true
  protocol_version: v3_canonical_no_leak
```

## Public Python API

```python
from gdkvm_project.models import build_model
from gdkvm_project.training import Trainer
from gdkvm_project.losses import LossComputer
from gdkvm_project.tracking import RunRecorder, MLflowLogger
from gdkvm_project.visualization import render_sequence_panel
```

Extension registries are available for future methods:

- `MODEL_REGISTRY`
- `LOSS_REGISTRY`
- `METRIC_COLLECTOR_REGISTRY`
- `VISUALIZER_REGISTRY`

## Tests

```bash
PYTHONPATH=src:. /home/tahara/miniconda3/bin/uv run pytest -q
```

The current test suite covers config composition, public registry boundaries,
GDKVM/DPFR synthetic smoke tests, local run recording, and visualization panel
rendering.

## Score Diagnostics

When a run scores lower than another server, compare artifacts before changing
the model:

```bash
PYTHONPATH=src:. /home/tahara/miniconda3/bin/uv run python -m gdkvm_project.reports.compare_runs /path/to/a /path/to/b
```

The most important fields are commit, config, data split counts, label sparsity,
effective batch size, backbone pretraining, AMP/CUDA/cuDNN, threshold, TTA, and
postprocess settings. DPFR should also be judged by anchor/prompt/flow deltas:
if anchor dice is low, refinement has little to rescue; if final-minus-anchor is
negative, prompt or flow refinement is hurting the segmentation task.

Current local evidence points to a few likely causes for lower DPFR scores:
unpretrained UNeXt anchors, small effective batch, sparse ED/ES supervision,
domain shift, and flow refinement that may not align well with low-texture
ultrasound boundaries.

For GDKVM-vs-DPFR reporting, the intended fairness grain is evaluation-facing:
both methods are compared on the same split, same visible video clip, same
`label_valid` supervised frames, same foreground Dice definition, and logits are
aligned to the target mask size before scoring. The methods may use different
internal temporal mechanisms as long as the test protocol and metric space stay
fixed.
