# Developing Methods And Modules

Future methods should integrate through registries instead of new branches in
the trainer or train entrypoint.

Minimum contract for a new method:

1. Add a method package under `src/gdkvm_project/models/<method>/`.
2. Register a builder in `MODEL_REGISTRY`.
3. Optionally register method losses in `LOSS_REGISTRY`.
4. Optionally register diagnostics in `METRIC_COLLECTOR_REGISTRY`.
5. Optionally register panels in `VISUALIZER_REGISTRY`.
6. Add `configs/model/<method>.yaml` and at least one synthetic smoke test.

Models should return tensors and aux dictionaries. They should not call MLflow
or write visualization files directly. Tracking and visualization consume model
outputs through the shared recorder, metric collector, and visualizer layers.
