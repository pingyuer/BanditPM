#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from dataset.frame_index import build_label_map


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
        has_first = 0 in valid_indices
        empty_masks = 0
        size_mismatch = 0
        for idx in valid_indices:
            mask = cv2.imread(str(label_dir / label_map[idx]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                issues["missing_mask_read"] += 1
                continue
            if size is not None and mask.shape != (size, size):
                size_mismatch += 1
            if int((mask > 0).sum()) == 0:
                empty_masks += 1
        rows.append(
            {
                "sample": img_dir.name,
                "frames": len(img_files),
                "labels": len(label_files),
                "valid_label_indices": valid_indices,
                "has_first_frame_gt": has_first,
                "empty_masks": empty_masks,
                "size_mismatch": size_mismatch,
                "metadata_protocol": metadata.get("protocol_name", ""),
            }
        )
    return {"rows": rows, "issues": issues}


def _summarize_camus(root: Path, split: str, size: int | None) -> dict:
    img_root = root / split / "img"
    label_root = root / split / "label"
    rows = []
    issues = Counter()
    for img_dir in sorted(img_root.glob("*")) if img_root.is_dir() else []:
        if not img_dir.is_dir():
            continue
        label_dir = label_root / img_dir.name
        img_files = sorted(p.name for p in img_dir.iterdir() if p.is_file())
        label_files = sorted(p.name for p in label_dir.iterdir() if p.is_file()) if label_dir.is_dir() else []
        valid_indices = []
        empty_masks = 0
        size_mismatch = 0
        for i, name in enumerate(label_files):
            if i >= len(img_files):
                break
            valid_indices.append(i)
            mask = cv2.imread(str(label_dir / name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                issues["missing_mask_read"] += 1
                continue
            if size is not None and mask.shape != (size, size):
                size_mismatch += 1
            if int((mask > 0).sum()) == 0:
                empty_masks += 1
        rows.append(
            {
                "sample": img_dir.name,
                "frames": len(img_files),
                "labels": len(label_files),
                "valid_label_indices": valid_indices,
                "has_first_frame_gt": 0 in valid_indices,
                "empty_masks": empty_masks,
                "size_mismatch": size_mismatch,
            }
        )
    return {"rows": rows, "issues": issues}


def _print_split(dataset: str, split: str, summary: dict, *, max_samples: int) -> None:
    rows = summary["rows"]
    frame_counts = Counter(row["frames"] for row in rows)
    label_counts = Counter(len(row["valid_label_indices"]) for row in rows)
    first_gt = sum(1 for row in rows if row["has_first_frame_gt"])
    empty = sum(row["empty_masks"] for row in rows)
    mismatch = sum(row["size_mismatch"] for row in rows)
    print(f"\n[{dataset}:{split}] samples={len(rows)}")
    print(f"  frame_count_distribution={dict(sorted(frame_counts.items()))}")
    print(f"  label_valid_distribution={dict(sorted(label_counts.items()))}")
    print(f"  samples_with_first_frame_gt={first_gt}/{len(rows)}")
    print(f"  empty_masks={empty} size_mismatch={mismatch} issues={dict(summary['issues'])}")
    if dataset == "echonet":
        sparse_like = sum(1 for row in rows if len(row["valid_label_indices"]) <= 2)
        print(f"  echonet_sparse_ed_es_like={sparse_like}/{len(rows)}")
    if dataset == "camus":
        dense_like = sum(1 for row in rows if len(row["valid_label_indices"]) == row["frames"])
        print(f"  camus_dense_like={dense_like}/{len(rows)}")
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
