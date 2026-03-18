#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify GDKVM-ready dataset folders.")
    parser.add_argument("--dataset", choices=["camus", "echonet"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def check_binary_mask(path: Path, errors: list[str]) -> None:
    values = set(np.unique(load_image(path)).tolist())
    if not values.issubset({0, 1}):
        errors.append(f"mask is not binary {{0,1}}: {path} -> {sorted(values)}")


def verify_camus(root: Path) -> list[str]:
    errors: list[str] = []
    split_path = root / "camus_public_datasplit_20250706.json"
    if not split_path.exists():
        errors.append(f"missing split json: {split_path}")
        return errors

    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    for key in ["train_data", "val_data", "test_data"]:
        if key not in split_data:
            errors.append(f"missing key in split json: {key}")

    all_patients = []
    for key in ["train_data", "val_data", "test_data"]:
        all_patients.extend(split_data.get(key, []))

    for patient_id in all_patients:
        img_dir = root / "img" / patient_id
        mask_dir = root / "gt_lv" / patient_id
        if not img_dir.is_dir():
            errors.append(f"missing image directory: {img_dir}")
            continue
        if not mask_dir.is_dir():
            errors.append(f"missing mask directory: {mask_dir}")
            continue

        img_files = sorted(img_dir.glob("*.png"))
        mask_files = sorted(mask_dir.glob("*.png"))
        if len(img_files) != 10:
            errors.append(f"{patient_id} image count != 10: {len(img_files)}")
        if len(mask_files) != 10:
            errors.append(f"{patient_id} mask count != 10: {len(mask_files)}")

        img_names = [path.name for path in img_files]
        mask_names = [path.name for path in mask_files]
        if img_names != mask_names:
            errors.append(f"{patient_id} image/mask filenames do not match")

        for path in img_files:
            if load_image(path).shape[:2] != (256, 256):
                errors.append(f"invalid image size for {path}")
        for path in mask_files:
            if load_image(path).shape[:2] != (256, 256):
                errors.append(f"invalid mask size for {path}")
            check_binary_mask(path, errors)

    return errors


def verify_echonet(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes = [
        "EchoNet verification checks file layout only.",
        "This dataset contract is endpoint-only weak supervision: labels should exist only for frames 0000 and 0009.",
        "A passing result does not prove that traced endpoints and sampled middle frames are semantically aligned.",
    ]
    for split in ["train", "val", "test"]:
        img_root = root / split / "img"
        label_root = root / split / "label"
        if not img_root.is_dir():
            errors.append(f"missing split image dir: {img_root}")
            continue
        if not label_root.is_dir():
            errors.append(f"missing split label dir: {label_root}")
            continue

        for case_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
            label_dir = label_root / case_dir.name
            if not label_dir.is_dir():
                errors.append(f"missing label dir for case: {case_dir.name}")
                continue

            img_files = sorted(case_dir.glob("*.png"))
            label_files = sorted(label_dir.glob("*.png"))
            if len(img_files) != 10:
                errors.append(f"{split}/{case_dir.name} image count != 10: {len(img_files)}")
            if len(label_files) != 2:
                errors.append(f"{split}/{case_dir.name} label count != 2: {len(label_files)}")
            if [path.name for path in label_files] != ["0000.png", "0009.png"]:
                errors.append(f"{split}/{case_dir.name} labels must be 0000.png and 0009.png")
            img_names = [path.name for path in img_files]
            if "0000.png" not in img_names or "0009.png" not in img_names:
                errors.append(f"{split}/{case_dir.name} is missing endpoint image frames 0000.png/0009.png")

            for path in img_files:
                if load_image(path).shape[:2] != (128, 128):
                    errors.append(f"invalid image size for {path}")
            for path in label_files:
                if load_image(path).shape[:2] != (128, 128):
                    errors.append(f"invalid label size for {path}")
                check_binary_mask(path, errors)

    return errors, notes


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    notes: list[str] = []
    if args.dataset == "camus":
        errors = verify_camus(root)
    else:
        errors, notes = verify_echonet(root)
    if errors:
        print("Verification failed:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    print(f"Verification passed for {args.dataset}: {root}")
    for note in notes:
        print(f"Note: {note}")


if __name__ == "__main__":
    main()
