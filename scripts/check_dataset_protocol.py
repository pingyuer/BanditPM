#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.frame_index import build_label_map, parse_frame_index


def _load_json(path: Path) -> dict:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _summarize_echonet(root: Path, split: str, size: int | None) -> dict:
    img_root = root / split / "img"
    label_root = root / split / "label"
    meta_root = root / split / "metadata"
    rows = []
    issues = Counter()
    for img_dir in sorted(img_root.glob("*")) if img_root.is_dir() else []:
        if not img_dir.is_dir():
            continue
        label_dir = label_root / img_dir.name
        img_files = sorted(p.name for p in img_dir.iterdir() if p.is_file())
        label_files = sorted(p.name for p in label_dir.iterdir() if p.is_file()) if label_dir.is_dir() else []
        metadata = _load_json(meta_root / f"{img_dir.name}.json")
        label_map = build_label_map(label_files, metadata, sample_name=img_dir.name)
        valid_indices = sorted(idx for idx in label_map if 0 <= idx < len(img_files))
        if not valid_indices:
            issues["label_valid_all_zero"] += 1
        has_first = 0 in valid_indices
        empty_masks = 0
        size_mismatch = 0
        image_shapes = []
        mask_shapes = []
        mask_unique_values = set()
        fg_ratios = []
        for idx in valid_indices:
            if 0 <= idx < len(img_files):
                image = cv2.imread(str(img_dir / img_files[idx]), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    issues["missing_image_read"] += 1
                else:
                    image_shapes.append(tuple(int(v) for v in image.shape))
                    if size is not None and image.shape != (size, size):
                        size_mismatch += 1
            mask = cv2.imread(str(label_dir / label_map[idx]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                issues["missing_mask_read"] += 1
                continue
            mask_shapes.append(tuple(int(v) for v in mask.shape))
            mask_unique_values.update(int(v) for v in np.unique(mask).tolist())
            if size is not None and mask.shape != (size, size):
                size_mismatch += 1
            if int((mask > 0).sum()) == 0:
                empty_masks += 1
            fg_ratios.append(float((mask > 0).mean()))
        source_frames = metadata.get("source_frames", [])
        source_to_local = {int(src): local for local, src in enumerate(source_frames)} if source_frames else {}
        ed_es_success = 0
        frame_source_success = 0
        unmapped_labels = 0
        for name in label_files:
            parsed = parse_frame_index(name, metadata)
            if parsed is None:
                unmapped_labels += 1
                continue
            upper_name = Path(name).stem.upper()
            if upper_name in {"ED", "ES"} and parsed in source_to_local:
                ed_es_success += 1
            if parsed in source_to_local and parse_frame_index(name, None) == parsed:
                frame_source_success += 1
        rows.append(
            {
                "sample": img_dir.name,
                "frames": len(img_files),
                "labels": len(label_files),
                "valid_label_indices": valid_indices,
                "has_first_frame_gt": has_first,
                "empty_masks": empty_masks,
                "size_mismatch": size_mismatch,
                "fg_ratios": fg_ratios,
                "image_shapes": image_shapes,
                "mask_shapes": mask_shapes,
                "mask_unique_values": sorted(mask_unique_values),
                "source_frames_len": len(source_frames),
                "source_frames_match_frames": len(source_frames) in {0, len(img_files)},
                "ed_es_mapping_success": ed_es_success,
                "frame_source_mapping_success": frame_source_success,
                "unmapped_labels": unmapped_labels,
                "metadata_protocol": metadata.get("protocol_name", ""),
            }
        )
    return {"rows": rows, "issues": issues}


def _summarize_camus(root: Path, split: str, size: int | None) -> dict:
    img_root = root / "img"
    label_root = root / "gt_lv"
    meta_root = root / "metadata"
    split_path = root / "camus_public_datasplit_20250706.json"
    split_key = {"train": "train_data", "val": "val_data", "test": "test_data"}.get(split, split)
    split_data = _load_json(split_path)
    patient_ids = split_data.get(split_key, []) if split_data else []
    rows = []
    issues = Counter()
    checked_paths = [str(img_root), str(label_root), str(split_path)]
    for patient_id in patient_ids:
        img_dir = img_root / patient_id
        if not img_dir.is_dir():
            issues["missing_img_dir"] += 1
            continue
        label_dir = label_root / patient_id
        if not label_dir.is_dir():
            issues["missing_mask_dir"] += 1
            continue
        img_files = sorted(p.name for p in img_dir.iterdir() if p.is_file())
        label_files = sorted(p.name for p in label_dir.iterdir() if p.is_file()) if label_dir.is_dir() else []
        valid_indices = []
        empty_masks = 0
        size_mismatch = 0
        image_shapes = []
        mask_shapes = []
        mask_unique_values = set()
        fg_ratios = []
        for i, name in enumerate(label_files):
            if i >= len(img_files):
                break
            valid_indices.append(i)
            image = cv2.imread(str(img_dir / img_files[i]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                issues["missing_image_read"] += 1
            else:
                image_shapes.append(tuple(int(v) for v in image.shape))
                if size is not None and image.shape != (size, size):
                    size_mismatch += 1
            mask = cv2.imread(str(label_dir / name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                issues["missing_mask_read"] += 1
                continue
            mask_shapes.append(tuple(int(v) for v in mask.shape))
            mask_unique_values.update(int(v) for v in np.unique(mask).tolist())
            if size is not None and mask.shape != (size, size):
                size_mismatch += 1
            if int((mask > 0).sum()) == 0:
                empty_masks += 1
            fg_ratios.append(float((mask > 0).mean()))
        if not valid_indices:
            issues["label_valid_all_zero"] += 1
        rows.append(
            {
                "sample": patient_id,
                "frames": len(img_files),
                "labels": len(label_files),
                "valid_label_indices": valid_indices,
                "has_first_frame_gt": 0 in valid_indices,
                "empty_masks": empty_masks,
                "size_mismatch": size_mismatch,
                "fg_ratios": fg_ratios,
                "image_shapes": image_shapes,
                "mask_shapes": mask_shapes,
                "mask_unique_values": sorted(mask_unique_values),
                "source_frames_len": len(_load_json(meta_root / f"{patient_id}.json").get("source_frames", [])),
            }
        )
    return {"rows": rows, "issues": issues, "checked_paths": checked_paths}


def _print_split(dataset: str, split: str, summary: dict, *, max_samples: int) -> None:
    rows = summary["rows"]
    frame_counts = Counter(row["frames"] for row in rows)
    label_counts = Counter(len(row["valid_label_indices"]) for row in rows)
    first_gt = sum(1 for row in rows if row["has_first_frame_gt"])
    empty = sum(row["empty_masks"] for row in rows)
    mismatch = sum(row["size_mismatch"] for row in rows)
    all_fg = [ratio for row in rows for ratio in row.get("fg_ratios", [])]
    image_shapes = sorted({shape for row in rows for shape in row.get("image_shapes", [])})
    mask_shapes = sorted({shape for row in rows for shape in row.get("mask_shapes", [])})
    mask_values = sorted({value for row in rows for value in row.get("mask_unique_values", [])})
    print(f"\n[{dataset}:{split}] samples={len(rows)}")
    print(f"  frame_count_distribution={dict(sorted(frame_counts.items()))}")
    print(f"  label_valid_distribution={dict(sorted(label_counts.items()))}")
    if image_shapes:
        print(f"  image_shape_unique={image_shapes}")
    if mask_shapes:
        print(f"  mask_shape_unique={mask_shapes}")
    if mask_values:
        print(f"  mask_unique_values={mask_values}")
    print(f"  samples_with_first_frame_gt={first_gt}/{len(rows)}")
    print(f"  empty_masks={empty} size_mismatch={mismatch} issues={dict(summary['issues'])}")
    if all_fg:
        print(
            "  mask_foreground_ratio="
            f"min={min(all_fg):.4f} mean={float(np.mean(all_fg)):.4f} "
            f"max={max(all_fg):.4f}"
        )
    if dataset == "echonet":
        sparse_like = sum(1 for row in rows if len(row["valid_label_indices"]) <= 2)
        print(f"  echonet_sparse_ed_es_like={sparse_like}/{len(rows)}")
        source_mismatch = sum(1 for row in rows if not row.get("source_frames_match_frames", True))
        ed_es_success = sum(row.get("ed_es_mapping_success", 0) for row in rows)
        frame_source_success = sum(row.get("frame_source_mapping_success", 0) for row in rows)
        unmapped = sum(row.get("unmapped_labels", 0) for row in rows)
        source_lens = Counter(row.get("source_frames_len", 0) for row in rows)
        print(f"  source_frames_len_distribution={dict(sorted(source_lens.items()))}")
        print(f"  source_frames_len_mismatch={source_mismatch}")
        print(f"  ed_es_mapping_success={ed_es_success} frame_source_mapping_success={frame_source_success} unmapped_labels={unmapped}")
    if dataset == "camus":
        dense_like = sum(1 for row in rows if len(row["valid_label_indices"]) == row["frames"])
        source_lens = Counter(row.get("source_frames_len", 0) for row in rows)
        print(f"  camus_dense_like={dense_like}/{len(rows)}")
        print(f"  source_frames_len_distribution={dict(sorted(source_lens.items()))}")
    if not rows:
        print(f"  checked_paths={summary.get('checked_paths', [])}")
    if summary["issues"].get("label_valid_all_zero", 0) > 0:
        raise SystemExit(f"{dataset}:{split} contains samples with label_valid all zero")
    for row in rows[:max_samples]:
        print(f"  sample={row['sample']} labels={row['valid_label_indices']} first_gt={row['has_first_frame_gt']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check video segmentation dataset protocol and sparse labels.")
    parser.add_argument("--data_path", default=str(Path(os.environ.get("DATASETS_ROOT", "~/datasets")) / "processed" / "echonet_png128_10f"))
    parser.add_argument("--dataset", choices=("echonet", "camus", "cardiacuda"), default="echonet")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=5)
    args = parser.parse_args()

    data_path = os.path.expanduser(os.path.expandvars(args.data_path))
    root = Path(data_path)
    print(f"data_path={root}")
    print(f"dataset={args.dataset}")
    for split in args.splits:
        if args.dataset in {"echonet", "cardiacuda"}:
            summary = _summarize_echonet(root, split, args.size)
        else:
            summary = _summarize_camus(root, split, args.size)
        _print_split(args.dataset, split, summary, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
