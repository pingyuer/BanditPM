# Architecture

The public architecture lives under `src/gdkvm_project/`.

## Boundaries

- `cli/`: train, eval, visualization, and future export entrypoints.
- `configs/`: config validation helpers; Hydra files live in repository `configs/`.
- `data/`: dataset registry facade and data utilities.
- `models/`: public model registry. Currently exposes only `gdkvm` and `dpfr`.
- `losses/`: common segmentation loss facade and method loss registry.
- `training/`: trainer, optimizer groups, hooks, checkpointing.
- `evaluation/`: evaluator facade and metric collector registry.
- `tracking/`: MLflow plus local `RunRecorder` artifacts.
- `visualization/`: sequence panels and method visualizer registry.
- `reports/`: reserved for experiment tables, comparison figures, and reports.

## Method Scope

`gdkvm` and `dpfr` are the only public method families. KPFF remains an internal
GDKVM component, not a public model name. DPFR owns its flow/grid helpers instead
of importing them from old method packages.

## Run Artifacts

Every run should have local artifacts independent of MLflow:

```text
config_resolved.yaml
runtime.json
git.json
metrics.jsonl
summary.json
data_flow_summary.json
```

MLflow remains the formal comparison surface for full experiments.

## Diagnostics

Use `gdkvm-compare` or `python -m gdkvm_project.reports.compare_runs` to compare
two run directories. The report highlights config, data-flow, runtime, git, and
summary differences that can invalidate a direct score comparison.

DPFR diagnostics are collected through `METRIC_COLLECTOR_REGISTRY` and visualized
with the DPFR panel. The key questions are:

- Is the anchor already weak?
- Does prompt refinement improve over anchor?
- Does flow refinement improve over prompt?
- Are flow magnitude, out-of-bound ratio, or fusion gates unusually large?

For sparse/domain echocardiography tasks, lower scores may reflect task-network
mismatch rather than an isolated bug: limited supervised frames, weak anchors,
domain shift, low-texture ultrasound boundaries, and flow-like refinement can
all reduce the usefulness of DPFR's full-window prompt/flow design.
