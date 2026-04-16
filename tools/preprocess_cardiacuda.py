#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


LOGGER = logging.getLogger("preprocess_cardiacuda")
DATASET_NAME = "cardiacuda_a4c_lv_png128_10f"
SITE_LABEL_MAP = {
    "Site_G_100": "train",
    "Site_R_126": "train",
    "Site_G_20": "val",
    "Site_R_52": "val",
    "Site_G_29": "test",
}
TARGET_LABEL_NAMES = {
    1: "lv",
    2: "la",
    3: "ra",
    4: "rv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess CardiacUDA A4C videos into the format required by GDKVM."
    )
    parser.add_argument("--input_root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("~/datasets/processed").expanduser(),
    )
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--target_label", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--supervision_mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense"],
        help="sparse=use Site_* sparse labels, dense=use label_all_frame dense labels.",
    )
    parser.add_argument(
        "--train_sites",
        type=str,
        default="Site_G_100,Site_R_126",
        help="Comma-separated CardiacUDA site folders used for train.",
    )
    parser.add_argument(
        "--val_sites",
        type=str,
        default="Site_G_20,Site_R_52",
        help="Comma-separated CardiacUDA site folders used for val.",
    )
    parser.add_argument(
        "--test_sites",
        type=str,
        default="Site_G_29",
        help="Comma-separated CardiacUDA site folders used for test.",
    )
    parser.add_argument("--dense_seed", type=int, default=42)
    parser.add_argument("--dense_val_count", type=int, default=2)
    parser.add_argument("--dense_test_count", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def parse_site_spec(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_cardiacuda_root(input_root: Path) -> Path:
    input_root = input_root.expanduser()
    candidates = [
        input_root / "CardiacUDA" / "full_extracted" / "cardiacUDC_dataset",
        input_root / "CardiacUDA" / "cardiacUDC_dataset",
        input_root / "full_extracted" / "cardiacUDC_dataset",
        input_root / "cardiacUDC_dataset",
        input_root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and (
            any((candidate / site).exists() for site in SITE_LABEL_MAP)
            or (candidate / "label_all_frame").is_dir()
        ):
            return candidate
    raise FileNotFoundError(f"Could not locate CardiacUDA root under {input_root}")


def scan_cardiacuda_structure(root: Path) -> dict[str, object]:
    site_summary: dict[str, dict[str, int]] = {}
    for site_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = len(list(site_dir.glob("*_image.nii.gz")))
        labels = len(list(site_dir.glob("*_label.nii.gz")))
        site_summary[site_dir.name] = {"images": images, "labels": labels}
    return {
        "source_root": str(root),
        "site_summary": site_summary,
        "notes": [
            "This pipeline currently targets A4C single-object segmentation only.",
            "Site_* folders contain sparse labels on a subset of frames.",
            "label_all_frame can be used for dense supervision, but some non-target labels may use a different encoding than Site_*.",
        ],
    }


def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros(volume.shape, dtype=np.uint8)
    scaled = (volume - vmin) / (vmax - vmin)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image).resize((size, size), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8), mode="L").resize(
        (size, size),
        resample=Image.Resampling.NEAREST,
    )
    mask_u8 = (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)
    if mask_u8.max() > 0 or np.max(mask) == 0:
        return mask_u8

    coords = np.argwhere(mask > 0)
    fallback = np.zeros((size, size), dtype=np.uint8)
    src_h, src_w = mask.shape
    y_idx = np.clip((coords[:, 0] * size) // max(src_h, 1), 0, size - 1)
    x_idx = np.clip((coords[:, 1] * size) // max(src_w, 1), 0, size - 1)
    fallback[y_idx, x_idx] = 1
    return fallback


def prepare_binary_mask(mask: np.ndarray) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask > 0)
    if filled.any():
        return filled.astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def build_sample_plan(label_frames: list[int], frame_count: int, num_frames: int) -> tuple[list[int], list[int]]:
    if frame_count < num_frames:
        raise ValueError(f"frame_count={frame_count} is smaller than num_frames={num_frames}")
    required = sorted(set(int(idx) for idx in label_frames))
    if len(required) > num_frames:
        raise ValueError(f"labelled frames exceed num_frames: {len(required)} > {num_frames}")

    chosen = set(required)
    window_start = required[0]
    window_end = frame_count - 1
    if len(chosen) < num_frames:
        chosen.add(window_end)

    dense_candidates = np.linspace(window_start, window_end, num_frames * 8)
    for idx in np.round(dense_candidates).astype(int).tolist():
        if len(chosen) >= num_frames:
            break
        chosen.add(int(idx))

    if len(chosen) < num_frames:
        for idx in range(window_start, frame_count):
            if len(chosen) >= num_frames:
                break
            chosen.add(idx)

    sampled_indices = sorted(chosen)
    if len(sampled_indices) > num_frames:
        optional = [idx for idx in sampled_indices if idx not in required]
        removable = len(sampled_indices) - num_frames
        sampled_indices = sorted(required + optional[:-removable])

    if len(sampled_indices) != num_frames:
        raise ValueError(
            f"failed to build a {num_frames}-frame plan from labels={label_frames} total={frame_count}"
        )
    label_positions = [sampled_indices.index(idx) for idx in required]
    return sampled_indices, label_positions


def build_dense_sample_plan(frame_count: int, num_frames: int) -> list[int]:
    if frame_count < num_frames:
        raise ValueError(f"frame_count={frame_count} is smaller than num_frames={num_frames}")
    indices = np.linspace(0, frame_count - 1, num_frames)
    return np.clip(np.round(indices).astype(int), 0, frame_count - 1).tolist()


def resolve_protocol_name(target_label: int, supervision_mode: str = "sparse") -> str:
    target_name = TARGET_LABEL_NAMES[target_label]
    return f"cardiacuda_a4c_{target_name}_{supervision_mode}"


def resolve_output_dataset_name(args: argparse.Namespace) -> str:
    target_label = int(getattr(args, "target_label", 1))
    image_size = int(getattr(args, "image_size", 128))
    target_name = TARGET_LABEL_NAMES[target_label]
    supervision_mode = str(getattr(args, "supervision_mode", "sparse"))
    suffix = "" if supervision_mode == "sparse" else f"_{supervision_mode}"
    return f"cardiacuda_a4c_{target_name}{suffix}_png{image_size}_{args.num_frames}f"


def ensure_simpleitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required to read CardiacUDA .nii.gz files.") from exc
    return sitk


def build_dense_split(case_names: list[str], seed: int, val_count: int, test_count: int) -> dict[str, list[str]]:
    import random

    if val_count < 0 or test_count < 0:
        raise ValueError("dense_val_count and dense_test_count must be non-negative")
    if len(case_names) <= val_count + test_count:
        raise ValueError(
            f"dense split requires train cases; got total={len(case_names)} val={val_count} test={test_count}"
        )

    shuffled = list(case_names)
    random.Random(seed).shuffle(shuffled)
    test_cases = sorted(shuffled[:test_count])
    val_cases = sorted(shuffled[test_count : test_count + val_count])
    train_cases = sorted(shuffled[test_count + val_count :])
    return {"train": train_cases, "val": val_cases, "test": test_cases}


def preprocess_dataset(args: argparse.Namespace) -> Path:
    sitk = ensure_simpleitk()
    root = resolve_cardiacuda_root(args.input_root)
    image_size = int(getattr(args, "image_size", 128))
    target_label = int(getattr(args, "target_label", 1))
    supervision_mode = str(getattr(args, "supervision_mode", "sparse"))
    train_sites = getattr(args, "train_sites", "Site_G_100,Site_R_126")
    val_sites = getattr(args, "val_sites", "Site_G_20,Site_R_52")
    test_sites = getattr(args, "test_sites", "Site_G_29")
    output_dir = (args.output_root / resolve_output_dataset_name(args)).expanduser()

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "preprocess.log")
    summary = scan_cardiacuda_structure(root)
    LOGGER.info("CardiacUDA directory summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))

    split_map: dict[str, list[str]]
    dense_case_split: dict[str, list[str]] | None = None
    if supervision_mode == "sparse":
        split_map = {
            "train": parse_site_spec(train_sites),
            "val": parse_site_spec(val_sites),
            "test": parse_site_spec(test_sites),
        }
    else:
        dense_root = root / "label_all_frame"
        if not dense_root.is_dir():
            raise FileNotFoundError(f"dense label root not found: {dense_root}")
        dense_case_names = sorted(
            path.stem.replace("_image.nii", "") for path in dense_root.glob("*_image.nii.gz")
        )
        dense_case_split = build_dense_split(
            dense_case_names,
            seed=int(getattr(args, "dense_seed", 42)),
            val_count=int(getattr(args, "dense_val_count", 2)),
            test_count=int(getattr(args, "dense_test_count", 2)),
        )
        split_map = {"train": ["label_all_frame"], "val": ["label_all_frame"], "test": ["label_all_frame"]}

    stats = {
        "processed": 0,
        "skipped": 0,
        "split_counts": Counter(),
        "site_counts": Counter(),
    }
    skip_reasons: Counter[str] = Counter()
    bad_cases: list[dict[str, object]] = []

    for split, sites in split_map.items():
        for site in sites:
            site_dir = root / site
            if not site_dir.is_dir():
                raise FileNotFoundError(f"Requested site directory not found: {site_dir}")

            image_paths = sorted(site_dir.glob("*_image.nii.gz"))
            for image_path in tqdm(image_paths, desc=f"CardiacUDA {split}:{site}"):
                label_path = image_path.with_name(image_path.name.replace("_image.nii.gz", "_label.nii.gz"))
                raw_case_name = image_path.stem.replace("_image.nii", "")
                case_name = f"{site.lower()}__{raw_case_name}"
                try:
                    if supervision_mode == "dense":
                        if dense_case_split is None:
                            raise ValueError("dense split not initialized")
                        if raw_case_name not in dense_case_split[split]:
                            continue

                    if not label_path.exists():
                        raise ValueError(f"missing {supervision_mode} label")

                    image_volume = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
                    label_volume = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
                    if image_volume.shape != label_volume.shape:
                        raise ValueError(
                            f"image/label shape mismatch: {image_volume.shape} vs {label_volume.shape}"
                        )
                    if image_volume.ndim != 3:
                        raise ValueError(f"expected 3D volume, found shape={image_volume.shape}")

                    target_mask = label_volume == target_label
                    labelled_frames = np.where(target_mask.reshape(target_mask.shape[0], -1).any(axis=1))[0].tolist()
                    if not labelled_frames:
                        raise ValueError(f"target label {target_label} is absent")

                    if supervision_mode == "sparse":
                        sampled_indices, label_positions = build_sample_plan(
                            labelled_frames,
                            frame_count=int(image_volume.shape[0]),
                            num_frames=args.num_frames,
                        )
                    else:
                        sampled_indices = build_dense_sample_plan(
                            frame_count=int(image_volume.shape[0]),
                            num_frames=args.num_frames,
                        )
                        label_positions = list(range(args.num_frames))

                    image_u8 = normalize_to_uint8(image_volume)
                    case_img_dir = output_dir / split / "img" / case_name
                    case_label_dir = output_dir / split / "label" / case_name
                    case_img_dir.mkdir(parents=True, exist_ok=True)
                    case_label_dir.mkdir(parents=True, exist_ok=True)

                    for out_idx, src_idx in enumerate(sampled_indices):
                        save_png(
                            resize_image(image_u8[src_idx], image_size),
                            case_img_dir / f"{out_idx:04d}.png",
                        )

                    label_source_frames = labelled_frames if supervision_mode == "sparse" else sampled_indices
                    for out_idx, src_idx in zip(label_positions, label_source_frames, strict=True):
                        binary_mask = resize_mask(
                            prepare_binary_mask(target_mask[src_idx].astype(np.uint8)),
                            image_size,
                        )
                        if binary_mask.max() == 0:
                            raise ValueError(f"empty mask after resize for frame {src_idx}")
                        save_png(binary_mask, case_label_dir / f"{out_idx:04d}.png")

                    sample_meta = {
                        "dataset": "cardiacuda",
                        "protocol_name": resolve_protocol_name(target_label, supervision_mode),
                        "num_frames": args.num_frames,
                        "label_indices": label_positions,
                        "source_frames": sampled_indices,
                        "sparse_source_frames": labelled_frames if supervision_mode == "sparse" else [],
                        "dense_source_frames": sampled_indices if supervision_mode == "dense" else [],
                        "target_label": target_label,
                        "target_name": TARGET_LABEL_NAMES[target_label],
                        "view": "A4C",
                        "split": split,
                        "site": site,
                        "case_name": case_name,
                        "source_case_name": raw_case_name,
                        "original_size": list(image_volume.shape[1:3]),
                        "resized_size": [image_size, image_size],
                        "supervision_mode": supervision_mode,
                    }
                    meta_dir = output_dir / split / "metadata"
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    (meta_dir / f"{case_name}.json").write_text(
                        json.dumps(sample_meta, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                    stats["processed"] += 1
                    stats["split_counts"][split] += 1
                    stats["site_counts"][site] += 1
                except Exception as exc:  # noqa: BLE001
                    stats["skipped"] += 1
                    reason = str(exc)
                    skip_reasons[reason] += 1
                    bad_cases.append(
                        {
                            "case_name": case_name,
                            "split": split,
                            "site": site,
                            "reason": reason,
                        }
                    )
                    LOGGER.exception("Failed to process CardiacUDA case %s", image_path.name)

    report = {
        "scan_summary": summary,
        "processed": stats["processed"],
        "skipped": stats["skipped"],
        "split_counts": dict(stats["split_counts"]),
        "site_counts": dict(stats["site_counts"]),
        "skip_reason_counts": dict(skip_reasons),
        "protocol_name": resolve_protocol_name(target_label, supervision_mode),
        "target_label": target_label,
        "target_name": TARGET_LABEL_NAMES[target_label],
        "ignored_inputs": ["Site_R_73"] if supervision_mode == "sparse" else ["Site_* sparse folders"],
        "task_contract": (
            "10 frames sampled from full clip with sparse supervision preserved at all annotated source frames."
            if supervision_mode == "sparse"
            else "10 frames sampled from full clip with dense supervision at every sampled frame."
        ),
        "supervision_mode": supervision_mode,
        "dense_case_split": dense_case_split,
    }
    (output_dir / "cardiacuda_bad_cases.json").write_text(
        json.dumps(bad_cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info("CardiacUDA preprocess summary:\n%s", json.dumps(report, indent=2, ensure_ascii=False))
    return output_dir


def main() -> None:
    args = parse_args()
    preprocess_dataset(args)


if __name__ == "__main__":
    main()
