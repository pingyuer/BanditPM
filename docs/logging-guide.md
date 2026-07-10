# Logging And Metrics

Formal experiment comparison uses MLflow. Local artifacts are still mandatory so
runs remain inspectable without a tracking server.

## Local Artifacts

`RunRecorder` owns the stable local files:

```text
config_resolved.yaml
runtime.json
git.json
metrics.jsonl
summary.json
data_flow_summary.json
```

Model code should not write these files directly. Training, evaluation, metric
collectors, and visualizers should pass structured data into recorder/tracking
facades.

## MLflow

`MLflowLogger` remains the remote tracking facade. Use tags for grouping, params
for compact comparable settings, metrics for scalar time series, and artifacts
for configs, checkpoints, visual panels, and environment/source records.

Recommended metric prefixes:

```text
train/...            training losses and learning rates
val/...              validation metrics
test/...             test metrics
gdkvm/...            GDKVM-specific diagnostics
dpfr/...             DPFR anchor/prompt/flow/fusion diagnostics
protocol/...         no-leak, init mode, frame validity
runtime/...          throughput and environment summaries
```

## Extension Flow

Future methods should expose diagnostics through `METRIC_COLLECTOR_REGISTRY` and
visual panels through `VISUALIZER_REGISTRY`. The trainer should call collectors
through stable interfaces instead of growing method-specific branches.
