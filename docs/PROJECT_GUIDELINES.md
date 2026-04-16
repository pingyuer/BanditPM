# Project Guidelines

## Goals
- Keep data contracts explicit across `dataset/`, `model/`, and `train.py`.
- Prefer small, testable helpers over embedding policy/config parsing inside long methods.
- Treat sparse-label supervision as a first-class mode, not a special-case afterthought.

## Module Boundaries
- `train.py`
  - owns orchestration only: dataset resolution, dataloader construction, trainer lifecycle.
  - should not embed supervision semantics or metric math.
- `dataset/`
  - owns sample contracts.
  - every dataset item should expose `rgb`, `cls_gt`, `label_valid`, and optionally `eval_valid`.
- `model/trainer.py`
  - owns train/eval loops, logging, checkpointing.
  - may consume frame-valid masks, but shared mask parsing should live in reusable helpers.
- `model/losses.py`
  - owns loss aggregation only.
  - must accept sample-level supervision masks so sparse protocols are representable without batch hacks.
- `utils/`
  - place shared, side-effect-light helpers here when logic is reused by more than one module.

## Testing Layout
- `tests/test_frame_validity.py`
  - pure unit tests for frame-valid mask normalization and summaries.
- `tests/test_supervision_indices.py`
  - trainer/loss-facing behavior tests.
- `tests/test_*_preprocess.py`
  - preprocessing contract tests.
- `tests/factories.py`
  - fake-data builders used by multiple tests.

## Fake-Data Strategy
- Prefer tensor-only fake batches for trainer/loss tests.
- Use tiny image sizes (`8x8`, `16x16`, `32x32`) and short sequences.
- Only hit filesystem in preprocessing/dataset tests.
- When testing sparse supervision, vary `label_valid` between samples inside the same batch.

## Refactor Policy
- Extract helpers when:
  - logic is reused in multiple modules,
  - semantics are easy to get wrong,
  - or tests benefit from isolating behavior.
- Avoid large architectural rewrites without first locking behavior in tests.
