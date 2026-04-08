# GDKVM Dataset Preprocessing

This document describes how to convert local `CAMUS_public` and `EchoNet-Dynamic` sources into the directory layouts expected by the current GDKVM dataset loaders in [dataset/vos_dataset.py](/home/tahara/GDKVM/dataset/vos_dataset.py) and [dataset/echo.py](/home/tahara/GDKVM/dataset/echo.py).

## What GDKVM expects

`TenCamusDataset` expects one root with:

```text
camus_png256_10f/
├── camus_public_datasplit_20250706.json
├── metadata/<patient_id>.json
├── img/<patient_id>/0000.png ... 0009.png
└── gt_lv/<patient_id>/0000.png ... 0009.png
```

`EchoDataset` expects one root with:

```text
echonet_png128_10f/
├── train/img/<case>/0000.png ... 0009.png
├── train/metadata/<case>.json
├── train/label/<case>/0000.png and 0009.png
├── val/img/<case>/...
├── val/label/<case>/...
├── test/img/<case>/...
└── test/label/<case>/...
```

Masks must stay binary `{0,1}` because the loaders explicitly test `mask == 1`.
For EchoNet, this is a sparse-label contract. `ed_to_es` remains endpoint-only, while `full_cycle` keeps sparse labels but expands the temporal span.

## Source structure recognized locally

### CAMUS

The local CAMUS tree is recognized as:

```text
~/datasets/CAMUS_public/
├── database_nifti/
│   ├── patient0001/
│   │   ├── Info_2CH.cfg
│   │   ├── Info_4CH.cfg
│   │   ├── patient0001_2CH_ED.nii.gz
│   │   ├── patient0001_2CH_ED_gt.nii.gz
│   │   ├── patient0001_2CH_ES.nii.gz
│   │   ├── patient0001_2CH_ES_gt.nii.gz
│   │   ├── patient0001_2CH_half_sequence.nii.gz
│   │   ├── patient0001_2CH_half_sequence_gt.nii.gz
│   │   ├── patient0001_4CH_ED.nii.gz
│   │   ├── patient0001_4CH_ED_gt.nii.gz
│   │   ├── patient0001_4CH_ES.nii.gz
│   │   ├── patient0001_4CH_ES_gt.nii.gz
│   │   ├── patient0001_4CH_half_sequence.nii.gz
│   │   └── patient0001_4CH_half_sequence_gt.nii.gz
│   └── ...
└── database_split/
    ├── subgroup_training.txt
    ├── subgroup_validation.txt
    └── subgroup_testing.txt
```

Behavior used by `preprocess_camus.py`:

- Only 4CH files are used.
- `Info_4CH.cfg` provides `ED`, `ES`, and `NbFrame`.
- `*_4CH_half_sequence.nii.gz` provides the image sequence.
- `*_4CH_half_sequence_gt.nii.gz` provides the segmentation sequence.
- `sampling_mode=short` samples `ED->ES`, while `sampling_mode=full` samples the full available half-sequence.
- Official split files are preserved when available.
- LV label defaults to `1`, which matches the public CAMUS convention. If a case contains only one positive label, the script falls back to that label automatically.

### EchoNet-Dynamic

The local EchoNet tree is recognized as:

```text
~/datasets/EchoNet-Dynamic/
├── FileList.csv
├── VolumeTracings.csv
└── Videos/
    ├── 0X100009310A3BD7FC.avi
    ├── 0X1002E8FBACD08477.avi
    └── ...
```

Behavior used by `preprocess_echonet.py`:

- Official split comes from `FileList.csv`, using the `Split` column.
- Tracings come from `VolumeTracings.csv`.
- Each traced case has two annotated frames.
- Each frame has contour point pairs `(X1, Y1)` and `(X2, Y2)`.
- The script rasterizes a polygon by concatenating the left contour with the reversed right contour.
- The output sequence always starts at the earlier traced frame and ends at the later traced frame.
- The script writes per-case metadata containing dataset, protocol name, label indices, source frames, and original/resized sizes.
- The script saves only two labels per clip; `ed_to_es` uses `0000/0009`, while `full_cycle` anchors the second label at the middle position.
- The script writes QA overlays to `qa_overlays/` and logs polygon warnings for suspicious contour geometry.
- Cases with missing tracing, missing video, empty mask, or invalid frame index are skipped and written to `echonet_bad_cases.json`.

## Dependencies

Python 3.10+ is assumed.

Install base packages:

```bash
pip install numpy pillow tqdm scipy opencv-python
```

Install CAMUS reader support:

```bash
pip install SimpleITK
```

## Generated files

- [preprocess_camus.py](/home/tahara/GDKVM/tools/preprocess_camus.py)
- [preprocess_echonet.py](/home/tahara/GDKVM/tools/preprocess_echonet.py)
- [preprocess_all.py](/home/tahara/GDKVM/tools/preprocess_all.py)
- [verify_gdkvm_dataset.py](/home/tahara/GDKVM/tools/verify_gdkvm_dataset.py)
- [run_preprocess.sh](/home/tahara/GDKVM/tools/run_preprocess.sh)

## Run preprocessing

Run both datasets:

```bash
python ~/GDKVM/tools/preprocess_all.py --input_root ~/datasets --output_root ~/datasets/processed --num_frames 10 --seed 42
```

Run CAMUS only:

```bash
python ~/GDKVM/tools/preprocess_camus.py --input_root ~/datasets --output_root ~/datasets/processed --num_frames 10 --seed 42
```

Run EchoNet only:

```bash
python ~/GDKVM/tools/preprocess_echonet.py --input_root ~/datasets --output_root ~/datasets/processed --num_frames 10 --seed 42
```

Overwrite existing processed folders:

```bash
python ~/GDKVM/tools/preprocess_all.py --input_root ~/datasets --output_root ~/datasets/processed --num_frames 10 --seed 42 --overwrite
```

## Verification

After preprocessing, verify both outputs:

```bash
python ~/GDKVM/tools/verify_gdkvm_dataset.py --dataset camus --root ~/datasets/processed/camus_png256_10f
python ~/GDKVM/tools/verify_gdkvm_dataset.py --dataset echonet --root ~/datasets/processed/echonet_png128_10f
```

`verify_gdkvm_dataset.py` validates layout and binary masks, but it does not guarantee that the sampled sequence preserves the original tracing semantics.

## Outputs and logs

Each processed dataset root stores:

- `preprocess.log`
- `camus_bad_cases.json` or `echonet_bad_cases.json`
- split statistics in the log output
- per-case warnings when repeated frames are needed

## Recommended paths

1. 处理后的推荐路径：
   CAMUS: `~/datasets/processed/camus_png256_10f`
   EchoNet: `~/datasets/processed/echonet_png128_10f`

2. 校验命令：
   `python ~/GDKVM/tools/verify_gdkvm_dataset.py --dataset camus --root ~/datasets/processed/camus_png256_10f`
   `python ~/GDKVM/tools/verify_gdkvm_dataset.py --dataset echonet --root ~/datasets/processed/echonet_png128_10f`

3. 一套最小运行命令：
   `python ~/GDKVM/tools/preprocess_all.py --input_root ~/datasets --output_root ~/datasets/processed --num_frames 10 --seed 42`
