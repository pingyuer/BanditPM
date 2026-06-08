# Experiment Matrix

## Canonical Methods

- `gdkvm`: original GDKVM-style shell with KPFF and original GDR memory.
- `kpff`: KPFF spatial fusion with temporal memory disabled.
- `unext_fusion`: UNeXt spatial backbone with DynaKey/mid-level memory fusion.

## Canonical Datasets

- `echo`: EchoNet endpoint clips, config suffix `_echo`.
- `camus`: CAMUS short dense clips, config suffix `_camus`.
- `domain`: CardiacUDA A4C LV dense domain data, config suffix `_domain`.

The nine recommended configs are:

```text
gdkvm_echo      gdkvm_camus      gdkvm_domain
kpff_echo       kpff_camus       kpff_domain
unext_fusion_echo  unext_fusion_camus  unext_fusion_domain
```

Run one config directly:

```bash
DATASETS_ROOT=$HOME/datasets \
/home/tahara/miniconda3/bin/uv run python train.py --config-name unext_fusion_camus
```

Run each method matrix:

```bash
bash scripts/run_canonical_matrix.sh
bash scripts/run_gdkvm_matrix.sh
bash scripts/run_kpff_matrix.sh
bash scripts/run_unext_fusion_matrix.sh
```

Run a sub-matrix with environment variables:

```bash
METHOD=unext_fusion DATASET=echo bash scripts/run_canonical_matrix.sh
METHOD=all DATASET=domain bash scripts/run_canonical_matrix.sh
```

## Outputs

Historical outputs are summarized into:

```text
outputs/EXPERIMENT_SUMMARY.csv
```

Regenerate and clean old run directories:

```bash
python scripts/summarize_and_clean_outputs.py --clean
```

Legacy scripts and configs are intentionally kept for old ablations, but new experiments should use the canonical configs above.

## Experimental Methods

`unext_ode_affine` is an experimental UNeXt-anchor method:

- UNeXt predicts the per-frame anchor logits.
- An online affine ODE slot bank keeps interpretable geometric state
  `[tx, ty, log_sx, log_sy, rot, shear]`.
- A low-resolution stationary velocity field optionally performs
  VoxelMorph-style proposal resampling with `grid_sample`.
- The final prediction keeps the FAF safety pattern: anchor/proposal trust
  fusion plus a small residual branch.

Entry configs:

```text
unext_ode_affine_echo
unext_ode_affine_camus
unext_ode_affine_domain
```
