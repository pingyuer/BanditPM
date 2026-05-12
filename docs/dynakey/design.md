# DynaKey: Reinforced Local ODE Memory Design

## Status

This document defines the first prototype design for **DynaKey: Reinforced Local ODE Memory**.

Current phase: **Step 1: Design only**.

No core implementation is included in this step. The next step is tests-first development after design confirmation.

## Goals

- Replace frame-cache-style temporal memory with a small dictionary of local ODE experts.
- Let each memory key represent local latent dynamics instead of one historical frame.
- Retrieve one or more ODE experts for the current latent state `z_t`.
- Predict a local next state `z_{t+1}^{pred}` through differentiable ODE-style dynamics.
- Provide a Q-learning maintenance policy for dictionary actions such as `keep`, `update`, `spawn`, `split`, and `delete`.
- Keep the first version modular, deterministic under fixed seeds, and easy to unit test.
- Integrate through the existing `MemoryCore` boundary without rewriting the segmentation pipeline.

## Non-Goals

- Full medical-performance optimization.
- A production-grade continuous ODE solver.
- Long-horizon reinforcement learning with replay buffer and target networks.
- Multi-object dictionary coupling beyond preserving the current `[B, N, ...]` object axis.
- Replacing KPFF, mask encoder, decoder, or object transformer.
- Changing existing dataset contracts such as `rgb`, `ff_gt`, `cls_gt`, `label_valid`, and `eval_valid`.
- Changing current GDR/BPM behavior.

## First-Version Scope

The first implementation should provide a minimal runnable prototype:

- `ODEKeyDictionary`
  - stores local ODE experts per batch item and object;
  - supports retrieval, one-step prediction, clone, spawn, update, split, and delete;
  - uses Euler integration for the first version;
  - exposes simple tensor diagnostics for logging.
- `DynaKeyQMaintainer`
  - builds Q-state tensors;
  - computes valid action masks;
  - returns action logits or Q-values;
  - supports rule or greedy selection for smoke integration;
  - provides counterfactual rollout return computation.
- `DynaKeyMemoryCore`
  - wraps dictionary retrieval and maintenance behind a `MemoryCore`-compatible call.
- Tests
  - unit tests for dictionary operations;
  - Q-state and action mask tests;
  - counterfactual return tests;
  - toy circle dynamics test;
  - minimal video clip integration smoke test.

## Deferred Work

- Higher-order ODE solvers such as RK4.
- Replay buffer, target Q-network, and double-DQN updates.
- Stochastic policy exploration schedules for long training runs.
- Dense multi-step supervision over unlabeled frames.
- Full checkpoint migration from existing GDR/BPM checkpoints.
- Learned uncertainty calibration for reward shaping.
- Large-scale benchmarking scripts.

## Core Data Structures

### Latent State

DynaKey should operate on latent states derived from existing model features.

First-version expected shape:

```text
z_BNCHW: [B, N, C, H, W]
```

Where:

- `B`: batch size
- `N`: number of objects
- `C`: latent channel dimension, expected to match `value_dim`
- `H, W`: low-resolution latent spatial size

For dictionary matching and Q-state construction, pooled states are used:

```text
z_BNC: [B, N, C]
```

The default pooling should be masked average pooling when a mask is available, otherwise spatial mean pooling.

### ODE Expert

Each dictionary slot stores one local dynamics expert:

```text
center_BNKC: [B, N, K, C]
velocity_BNKC: [B, N, K, C]
scale_BNK: [B, N, K]
age_BNK: [B, N, K]
usage_BNK: [B, N, K]
error_ema_BNK: [B, N, K]
valid_BNK: [B, N, K]
```

Where `K` is dictionary size per object.

Interpretation:

- `center`: local latent region where the expert is valid.
- `velocity`: first-order local ODE vector field value around the center.
- `scale`: retrieval temperature or radius.
- `age`: number of maintenance steps since slot creation.
- `usage`: retrieval count or soft usage mass.
- `error_ema`: exponential moving average of prediction error.
- `valid`: slot occupancy.

First-version local dynamics:

```text
f_k(z_t) = velocity_k
z_{t+1,k}^{pred} = z_t + dt * f_k(z_t)
z_{t+1}^{pred} = sum_k retrieval_weight_k * z_{t+1,k}^{pred}
```

This is intentionally simple. It can later be upgraded to affine local dynamics:

```text
f_k(z_t) = A_k (z_t - center_k) + b_k
```

### Dictionary State Object

Implementation should use a small state container:

```python
@dataclass
class ODEKeyDictionaryState:
    center: torch.Tensor
    velocity: torch.Tensor
    scale: torch.Tensor
    age: torch.Tensor
    usage: torch.Tensor
    error_ema: torch.Tensor
    valid: torch.Tensor
```

State should be reset per video, matching current `MemoryCore.reset_state(...)` behavior.

## ODE Key Dictionary Interface

Proposed module:

```text
model/modules/dynakey/
  __init__.py
  ode_key_dictionary.py
  q_maintainer.py
  counterfactual.py
  dynakey_memory_core.py
```

### `ODEKeyDictionary`

Constructor:

```python
ODEKeyDictionary(
    value_dim: int,
    bank_size: int,
    dt: float = 1.0,
    ema_alpha: float = 0.2,
    retrieval_temperature: float = 1.0,
    min_scale: float = 1e-3,
)
```

Required methods:

```python
reset_state(batch_size: int, num_objects: int, device: torch.device) -> None
state -> ODEKeyDictionaryState
retrieve(z_BNC: torch.Tensor) -> tuple[torch.Tensor, dict]
predict(z_BNC: torch.Tensor, weights_BNK: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]
clone_slot(src_idx: torch.Tensor, dst_idx: torch.Tensor) -> None
spawn(z_BNC: torch.Tensor, velocity_BNC: torch.Tensor | None = None, slot_idx: torch.Tensor | None = None) -> torch.Tensor
update(z_BNC: torch.Tensor, target_next_BNC: torch.Tensor | None, slot_idx: torch.Tensor, weight: torch.Tensor | None = None) -> None
split(slot_idx: torch.Tensor, perturb_scale: float = 0.01) -> torch.Tensor
delete(slot_idx: torch.Tensor) -> None
diagnostics() -> dict
```

Return conventions:

- `retrieve` returns `weights_BNK` and aux values such as `nearest_idx`, `distance`, `occupancy`.
- `predict` returns `z_next_pred_BNC` and aux values such as `slot_pred_BNKC`, `weights_BNK`.
- mutating methods update internal state in-place but should avoid breaking autograd for model inputs.

### Retrieval

Retrieval should compute masked similarity:

```text
distance_k = || normalize(z_t) - normalize(center_k) ||_2
logit_k = -distance_k / temperature
invalid slots get -inf
weights = softmax(logits)
```

If no slot is valid for a sample/object:

- return zero weights;
- set `has_match=False`;
- let caller decide whether to spawn.

### Prediction

First-version prediction uses Euler:

```text
slot_pred_k = z_t + dt * velocity_k
z_next_pred = weighted_sum(slot_pred_k)
```

Fallback when no slot is valid:

```text
z_next_pred = z_t
```

This fallback should be explicit in aux as `used_identity_fallback`.

## Q-Learning Maintainer Interface

### `DynaKeyQMaintainer`

Constructor:

```python
DynaKeyQMaintainer(
    value_dim: int,
    bank_size: int,
    hidden_dim: int = 256,
    num_actions: int = 5,
    gamma: float = 0.95,
)
```

Required methods:

```python
build_q_state(
    z_BNC: torch.Tensor,
    z_next_pred_BNC: torch.Tensor,
    target_next_BNC: torch.Tensor | None,
    dictionary_state: ODEKeyDictionaryState,
    retrieval_aux: dict,
) -> torch.Tensor

action_mask(dictionary_state: ODEKeyDictionaryState, retrieval_aux: dict) -> torch.Tensor
forward(q_state: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor
select_action(q_values: torch.Tensor, action_mask: torch.Tensor, mode: str = "greedy") -> torch.Tensor
```

The first version should expose Q-values but does not need to run full DQN training inside the segmentation loop.

## Action Space

First-version action ids:

```text
0: keep
1: update
2: spawn
3: split
4: delete
```

### Action Semantics

- `keep`
  - no dictionary mutation except passive `age += 1`.
- `update`
  - update nearest or selected slot center and velocity using EMA.
- `spawn`
  - create a new slot from current state if an empty slot exists; if full, use configured fallback.
- `split`
  - duplicate a high-error or selected slot with small perturbation.
- `delete`
  - invalidate a selected slot, usually stale or high-error.

### Validity Rules

Required action mask behavior:

- `keep` is always valid.
- `update` is valid only when at least one slot is valid.
- `spawn` is valid when at least one slot is empty.
- `split` is valid when at least one slot is valid and at least one empty slot exists.
- `delete` is valid when at least one slot is valid.

The mask shape should be:

```text
action_mask_BNA: [B, N, 5]
```

Invalid actions should be masked to a large negative value in Q selection.

## State Representation

Q-state should be compact and numerically stable.

Proposed per sample/object vector:

```text
q_state = concat(
    z_t,
    z_next_pred - z_t,
    prediction_error_features,
    retrieval_features,
    dictionary_global_features,
    selected_slot_features
)
```

First-version components:

- `z_t`: `[C]`
- `delta_pred`: `[C]`
- `prediction_error_features`: `[3]`
  - L2 error if target exists, else `0`
  - cosine error if target exists, else `0`
  - `has_target` flag
- `retrieval_features`: `[4]`
  - max retrieval weight
  - entropy
  - has valid match
  - nearest distance
- `dictionary_global_features`: `[5]`
  - occupancy ratio
  - mean age
  - mean usage
  - mean error EMA
  - max error EMA
- `selected_slot_features`: `[4]`
  - selected slot age
  - selected slot usage
  - selected slot error EMA
  - selected slot valid flag

Total first-version dimension:

```text
2 * C + 16
```

All scalar features should be finite, clamped where needed, and normalized by safe denominators.

## Counterfactual Rollout Training

The goal is to evaluate maintenance actions without needing to commit each action to the live dictionary.

### Inputs

```python
compute_counterfactual_returns(
    dictionary: ODEKeyDictionary,
    z_t_BNC: torch.Tensor,
    z_tp1_BNC: torch.Tensor,
    actions_BN: torch.Tensor,
    rewards_cfg: dict,
) -> tuple[torch.Tensor, dict]
```

### Procedure

For each candidate action:

1. Clone dictionary state.
2. Apply the candidate action to the cloned state.
3. Predict `z_{t+1}` from `z_t`.
4. Compute reward:

```text
reward = -prediction_error - action_cost + stability_bonus
```

First-version prediction error:

```text
prediction_error = mean((z_pred - z_target)^2)
```

Action costs:

```text
keep: 0.00
update: 0.02
spawn: 0.05
split: 0.08
delete: 0.04
```

5. Return one-step target:

```text
return = reward
```

The first version should not bootstrap from a target Q-network. This keeps the rollout deterministic and testable.

### Output

```text
returns_BNA: [B, N, A]
```

Aux diagnostics:

- per-action prediction error;
- per-action cost;
- best action;
- finite mask.

## Integration With Existing Segmentation Pipeline

### Existing Boundary

The current segmentation model already routes temporal behavior through:

```python
MemoryCore.forward(
    value_BNCHW,
    key_BCHW,
    pixfeat_BCHW,
    mask_BNHW,
    policy_meta,
)
```

DynaKey should be integrated by adding a new memory type:

```yaml
model:
  memory_core:
    type: "dynakey"
  temporal_memory:
    type: "dynakey"
```

### Proposed First Integration

Inside `MemoryCore`:

- instantiate `DynaKeyMemoryCore` when memory type is `dynakey`;
- call it from `MemoryCore.forward`;
- return `readout_BNCHW` with the same shape as `value_BNCHW`;
- expose aux under `dynakey_aux`.

### Latent Mapping

First version:

- pool `value_BNCHW` to `z_BNC`;
- use dictionary to predict `z_next_pred_BNC`;
- broadcast predicted delta back to spatial map:

```text
delta_BNC = z_next_pred_BNC - z_BNC
readout_BNCHW = value_BNCHW + gate * delta_BNC[..., None, None]
```

This preserves the existing decoder path and avoids changing mask decoder inputs.

### Target for Dictionary Update

During frame `t`, the model does not yet know the encoded value for frame `t+1` inside `MemoryCore.forward`.

First-version update target:

- use current transition proxy from previous call:
  - store previous `z_{t-1}`;
  - when current `z_t` arrives, update previous prediction error;
  - then maintain dictionary using current `z_t`.

This mirrors per-video recurrent state and avoids changing `GDKVM.forward` loop signatures.

### Aux Logging

DynaKey should emit:

```text
dynakey/occupancy_ratio
dynakey/retrieval_entropy
dynakey/prediction_error
dynakey/action_keep
dynakey/action_update
dynakey/action_spawn
dynakey/action_split
dynakey/action_delete
```

Example log row:

```text
[DynaKey] iter=120 action_update=0.54 action_spawn=0.18 occupancy=0.62 pred_mse=0.031
```

## Failure Modes And Guards

### Empty Dictionary

Risk:

- no valid experts, retrieval is undefined.

Guard:

- identity prediction fallback;
- `spawn` valid when empty;
- aux flag `used_identity_fallback`.

### Slot Collapse

Risk:

- all centers become similar and dictionary loses coverage.

Guard:

- split high-error slots;
- spawn into empty slots;
- optional center diversity penalty later.

### Exploding Dynamics

Risk:

- velocity grows too large and destabilizes segmentation.

Guard:

- clamp velocity norm;
- clamp predicted delta norm;
- initialize velocity to zero;
- expose finite checks in tests.

### Invalid Action Selection

Risk:

- Q network chooses impossible maintenance action.

Guard:

- action masks in `select_action`;
- tests enforce invalid actions cannot be selected.

### Autograd Corruption Through State Mutation

Risk:

- recurrent dictionary mutation stores graph tensors across frames.

Guard:

- dictionary state stores detached tensors;
- prediction path may use differentiable `z_t`, but persistent state is detached.

### Sparse Label Mismatch

Risk:

- DynaKey reward relies on unavailable ground-truth frames.

Guard:

- first version uses latent transition consistency, not segmentation labels;
- segmentation loss still follows existing `label_valid`.

### DDP Inconsistency

Risk:

- per-rank recurrent dictionary state diverges across samples.

Guard:

- state is per batch on each rank, reset per video batch;
- no cross-rank state synchronization in first version.

## Test Plan

Tests should be written before implementation.

### A. `ODEKeyDictionary` Unit Tests

File:

```text
tests/test_dynakey_ode_dictionary.py
```

Cases:

- initializes tensors with expected shapes and all slots invalid;
- `spawn` creates valid slots with finite centers and velocities;
- `retrieve` returns valid weights summing to one when slots exist;
- `retrieve` handles empty dictionary with zero weights and fallback aux;
- `predict` returns identity when no valid slots exist;
- `predict` returns Euler prediction with known velocity;
- `clone_slot` duplicates source slot values;
- `update` changes selected slot center, velocity, usage, and error EMA;
- `split` creates a second valid nearby slot;
- `delete` invalidates a slot and clears diagnostics.

### B. Q-State Tests

File:

```text
tests/test_dynakey_q_state.py
```

Cases:

- `build_q_state` returns shape `[B, N, 2*C + 16]`;
- all values are finite;
- missing target sets `has_target=0`;
- present target sets `has_target=1` and non-negative error features;
- retrieval entropy is finite with empty and non-empty dictionaries.

### C. Action Mask Tests

File:

```text
tests/test_dynakey_action_mask.py
```

Cases:

- empty dictionary allows `keep` and `spawn` only;
- full dictionary disallows `spawn` and `split`;
- partially filled dictionary allows all valid maintenance actions;
- invalid actions are never selected by greedy selection.

### D. Counterfactual Rollout Tests

File:

```text
tests/test_dynakey_counterfactual.py
```

Cases:

- returns shape `[B, N, A]`;
- all returns are finite;
- action costs lower return when predictions are equal;
- update or spawn improves return for a simple known transition;
- live dictionary state is unchanged after counterfactual rollout.

### E. Toy Circle Dynamics Test

File:

```text
tests/test_dynakey_toy_circle.py
```

Scenario:

- generate 2D latent states moving around a unit circle;
- true local dynamics approximates tangential velocity;
- spawn/update dictionary over a short sequence;
- prediction error after maintenance is lower than identity baseline.

This test prevents hard-coded static behavior while staying small.

### F. Minimal Clip Integration Smoke Test

File:

```text
tests/test_dynakey_minimal_integration.py
```

Cases:

- instantiate `MemoryCore` with `memory_core.type=dynakey`;
- pass a tiny fake clip-like sequence through repeated memory calls;
- output readout shape equals input `value_BNCHW`;
- aux contains `memory_type=dynakey` and `dynakey_aux`;
- all readout values are finite;
- no existing `original_gdr`, `none`, or `bpm` path is modified.

## Acceptance Criteria

The task is complete only when all of the following pass:

- ODEKeyDictionary unit tests
- QNetwork / q_state unit tests
- action mask tests
- counterfactual rollout tests
- toy circle dynamics test
- minimal EchoNet/CAMUS clip integration smoke test

Final delivery must include:

- this design document path;
- new or modified source files;
- test file list;
- exact test command;
- test result summary;
- key log example;
- explicit disclosure of any failing test if present.
