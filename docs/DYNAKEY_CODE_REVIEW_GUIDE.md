# DynaKey Code Review Guide

This document is a practical review map for the current DynaKey prototype implementation.

It explains where to start reading, what files were changed, which contracts matter most, and what reviewers should verify before trusting the experiment results.

## Review Entry Points

Start from the existing training and model pipeline:

1. `train.py`
   - Hydra entry point.
   - Builds datasets and loaders.
   - Instantiates `Trainer`.
   - Does not contain DynaKey-specific logic.

2. `model/trainer.py`
   - Runs training/evaluation.
   - Calls `self.model(data)` in `Trainer.do_pass`.
   - Logs DynaKey statistics through `_log_dynakey_stats`.

3. `model/gdkvm01.py`
   - Main segmentation model.
   - Calls `self.memory_core(...)` once per frame.
   - DynaKey enters the model through this existing memory boundary.

4. `model/modules/memory_core.py`
   - Main integration switch.
   - New `memory_type == "dynakey"` branch instantiates and calls `DynaKeyMemoryCore`.
   - This is the most important integration file to review first.

## Main DynaKey Files

Core implementation:

- `model/modules/dynakey/ode_key_dictionary.py`
  - ODE expert dictionary.
  - Stores per-video local dynamics slots.
  - Implements `retrieve`, `predict`, `spawn`, `update`, `split`, `delete`.

- `model/modules/dynakey/q_maintainer.py`
  - Q-state builder.
  - Action mask logic.
  - Lightweight Q-network scaffold and action selection.
  - The Q-network is not trained by the main segmentation loss yet.

- `model/modules/dynakey/counterfactual.py`
  - One-step counterfactual rollout utility.
  - Used by tests and future policy training work.

- `model/modules/dynakey/dynakey_memory_core.py`
  - Runtime wrapper used by the segmentation model.
  - Converts dense feature maps into pooled latent states.
  - Calls the ODE dictionary.
  - Returns a `readout_BNCHW` tensor compatible with the existing decoder path.

- `model/modules/dynakey/__init__.py`
  - Public exports for DynaKey modules.

Integration files:

- `model/modules/memory_core.py`
  - Adds `dynakey` memory type.
  - Freezes original GDR parameters for this path.
  - Returns DynaKey aux data under `dynakey_aux`.

- `model/modules/__init__.py`
  - Exposes DynaKey symbols.

- `model/trainer.py`
  - Adds `_log_dynakey_stats`.
  - Logs occupancy, retrieval entropy, identity fallback, and action histogram.

Experiment scripts:

- `scripts/run_dynakey_gpu0.sh`
  - DynaKey queue for GPU 0.

- `scripts/run_dynakey_gpu1.sh`
  - DynaKey queue for GPU 1.

Design document:

- `docs/DYNAKEY_DESIGN.md`
  - Higher-level design and test plan.

## Runtime Data Flow

DynaKey does not replace the segmentation decoder. It only modifies the temporal memory readout.

Current frame-level flow:

```text
GDKVM.forward
  -> encode image and mask
  -> value_BNCHW, key_BCHW, pixfeat_BCHW, mask_BNHW
  -> MemoryCore.forward
  -> DynaKeyMemoryCore.forward
  -> readout_BNCHW
  -> pixel_fusion
  -> object_transformer
  -> mask_decoder
  -> logits/masks
```

Inside `DynaKeyMemoryCore.forward`:

```text
value_BNCHW
  -> mask-aware pooling
  -> z_BNC
  -> compute previous-prediction-vs-current residual
  -> select and execute a maintenance action
  -> retrieve ODE experts after maintenance
  -> spawn if dictionary is empty or policy requests spawn
  -> predict z_next_pred_BNC
  -> delta = z_next_pred_BNC - z_BNC
  -> readout_BNCHW = value_BNCHW + gate * delta[..., None, None]
```

Persistent per-video state:

```text
ODEKeyDictionaryState
  center:    [B, N, K, C]
  velocity:  [B, N, K, C]
  scale:     [B, N, K]
  age:       [B, N, K]
  usage:     [B, N, K]
  error_ema: [B, N, K]
  valid:     [B, N, K]
```

Important shape contract:

```text
input  value_BNCHW: [B, N, C, H, W]
output readout_BNCHW: [B, N, C, H, W]
```

If this shape contract breaks, the downstream decoder path will fail.

## Current Prototype Scope

Implemented:

- local Euler ODE expert prediction;
- dictionary slot lifecycle operations;
- mask-aware latent pooling;
- one-step previous-to-current latent update;
- live execution of selected maintenance actions;
- true previous-prediction-vs-current-latent prediction error logging;
- scale-aware retrieval;
- residual-aware split;
- Q-state construction;
- action masks;
- Q-value computation;
- greedy action selection for logging/prototype behavior;
- counterfactual rollout helper;
- MemoryCore integration;
- tests and smoke integration.

Not fully implemented yet:

- trained Q-learning policy in the main training loss;
- replay buffer;
- target Q-network;
- multi-step rollout objective;
- segmentation-label-based DynaKey reward;
- affine or neural ODE dynamics.

The current Q-maintainer is a scaffold plus diagnostics path. In `q_greedy` mode selected actions are executed in the live dictionary, but the Q-network itself is not trained by a Q loss in the main training loop.

## Key Review Questions

### 1. MemoryCore Integration

File:

```text
model/modules/memory_core.py
```

Review:

- Does `memory_type == "dynakey"` resolve correctly from Hydra config?
- Does `reset_state(...)` reset DynaKey state per video batch?
- Does `forward(...)` return `readout_BNCHW` and aux without changing other memory modes?
- Are GDR/BPM/none branches untouched behaviorally?

### 2. Dictionary Mutation Safety

File:

```text
model/modules/dynakey/ode_key_dictionary.py
```

Review:

- Persistent dictionary tensors should store detached values.
- `spawn` and `update` should not keep full autograd graphs across frames.
- Empty dictionary retrieval should return zero weights and identity fallback.
- Invalid slots must not receive retrieval probability.
- `delete` should clear slot values and validity only when more than one key remains.
- Velocity should be clamped to avoid exploding dynamics.

### 3. Shape and Numerical Stability

Files:

```text
model/modules/dynakey/ode_key_dictionary.py
model/modules/dynakey/q_maintainer.py
model/modules/dynakey/dynakey_memory_core.py
```

Review:

- No hidden assumption that `B == 1` or `N == 1`, even though current datasets mostly use one object.
- `B, N, K, C` tensors stay aligned.
- `torch.nan_to_num` and clamps are used where divisions/logs occur.
- Action masking uses fp16-safe values. Current invalid fill is `-1e4`, not `-1e9`, because AMP half precision overflows on `-1e9`.

### 4. DynaKey Readout Semantics

File:

```text
model/modules/dynakey/dynakey_memory_core.py
```

Review:

- `z_BNC` is pooled from `value_BNCHW`.
- Predicted latent delta is broadcast back to `[B, N, C, H, W]`.
- The output is:

```python
readout = value_BNCHW + gate * delta
```

- This keeps the existing decoder path unchanged.

### 5. Logging

File:

```text
model/trainer.py
```

Review:

- `_log_dynakey_stats` is best-effort and wrapped safely.
- DynaKey logs should not affect training if aux keys are missing.
- Expected scalar names:

```text
dynakey/occupancy_ratio
dynakey/active_key_count
dynakey/retrieval_entropy
dynakey/identity_fallback
dynakey/prediction_error
dynakey/residual_norm
dynakey/action_keep
dynakey/action_update
dynakey/action_spawn
dynakey/action_split
dynakey/action_delete
```

### 6. Experiment Scripts

Files:

```text
scripts/run_dynakey_gpu0.sh
scripts/run_dynakey_gpu1.sh
```

Review:

- `model.memory_core.type=dynakey`
- `model.temporal_memory.type=dynakey`
- DynaKey config is added under `+model.memory_core.dynakey.*`
- W&B mode is `online`.
- Logs go to `outputs/BanditPM/tmux_logs`.

## Tests To Review

DynaKey tests:

- `tests/test_dynakey_ode_dictionary.py`
  - dictionary init/retrieve/predict/spawn/update/split/delete.

- `tests/test_dynakey_q_state.py`
  - Q-state shape and finite values.

- `tests/test_dynakey_action_mask.py`
  - valid action masks and masked greedy action selection.

- `tests/test_dynakey_counterfactual.py`
  - counterfactual returns and no live-state mutation.

- `tests/test_dynakey_toy_circle.py`
  - toy 2D circle dynamics beats identity baseline.

- `tests/test_dynakey_minimal_integration.py`
  - MemoryCore smoke test for `memory_core.type=dynakey`.

Recommended test command:

```bash
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run pytest -q tests/test_dynakey_ode_dictionary.py tests/test_dynakey_q_state.py tests/test_dynakey_action_mask.py tests/test_dynakey_counterfactual.py tests/test_dynakey_toy_circle.py tests/test_dynakey_minimal_integration.py
```

Full regression:

```bash
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run pytest -q
```

Last verified result during implementation:

```text
35 passed, 3 warnings
```

## Config Review

DynaKey currently uses CLI overrides rather than a dedicated YAML config.

Example:

```bash
model.memory_core.type=dynakey
model.temporal_memory.type=dynakey
+model.memory_core.dynakey.BANK_SIZE=4
+model.memory_core.dynakey.DT=1.0
+model.memory_core.dynakey.EMA_ALPHA=0.2
+model.memory_core.dynakey.RETRIEVAL_TEMPERATURE=1.0
+model.memory_core.dynakey.HIDDEN_DIM=256
+model.memory_core.dynakey.GATE_INIT=1.0
+model.memory_core.dynakey.POLICY_MODE=fixed_residual
+model.memory_core.dynakey.FORCED_ACTION=spawn
+model.memory_core.dynakey.RESIDUAL_SPAWN_THRESHOLD=0.05
+model.memory_core.dynakey.MIN_SCALE=0.001
+model.memory_core.dynakey.SPLIT_EPS=0.01
+model.memory_core.dynakey.SPLIT_SCALE_FACTOR=0.7
```

Reviewers may want to turn this into a dedicated config file later, for example:

```text
config/config_dynakey_baseline.yaml
```

## Known Risks

- Q-learning is not fully trained yet; the current policy path is mostly infrastructure.
- Dense CardiacUDA DynaKey performance is weak in current experiments.
- Dictionary dynamics are first-order Euler with constant velocity per slot.
- This is a first-order Euler local dynamics prototype, not a full local linear ODE operator.
- The update target is latent transition consistency, not direct mask quality.
- The dictionary is reset per video batch and is not a global memory across dataset samples.
- Current implementation has not been tuned for throughput; GPU utilization is modest.

## Suggested Review Order

1. Read `docs/DYNAKEY_DESIGN.md` for intent.
2. Read `model/modules/memory_core.py` to understand integration.
3. Read `model/modules/dynakey/dynakey_memory_core.py` for live data flow.
4. Read `model/modules/dynakey/ode_key_dictionary.py` for state mutation correctness.
5. Read `model/modules/dynakey/q_maintainer.py` and `counterfactual.py`.
6. Read tests under `tests/test_dynakey_*.py`.
7. Check scripts under `scripts/run_dynakey_gpu*.sh`.
8. Compare W&B results against baseline runs.

## Quick Mental Model

DynaKey is currently:

```text
existing GDKVM segmentation pipeline
  + a DynaKey memory readout
  + per-video local ODE dictionary
  + latent transition updates
  + Q-policy scaffolding
```

It is not yet:

```text
full DQN-trained memory maintenance
or a replacement segmentation architecture
or a label-driven temporal RL system
```
