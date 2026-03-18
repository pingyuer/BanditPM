#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


LOGGER = logging.getLogger("preprocess_camus")
DATASET_NAME = "camus_png256_10f"
SPLIT_JSON_NAME = "camus_public_datasplit_20250706.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess CAMUS into the format required by GDKVM."
    )
    parser.add_argument("--input_root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("~/datasets/processed").expanduser(),
    )
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--lv_label",
        type=int,
        default=1,
        help="CAMUS LV label value. Defaults to the official convention: 1.",
    )
    return parser.parse_args()


def ensure_simpleitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required to read CAMUS .nii.gz files. "
            "Install it with: pip install SimpleITK"
        ) from exc
    return sitk


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


def read_cfg(cfg_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def scan_camus_structure(camus_root: Path) -> dict[str, object]:
    db_root = camus_root / "database_nifti"
    split_root = camus_root / "database_split"
    patients = sorted(p for p in db_root.iterdir() if p.is_dir() and p.name.startswith("patient"))
    sample_patients = [p.name for p in patients[:5]]
    sample_files = {
        p.name: sorted(item.name for item in p.iterdir())
        for p in patients[:3]
    }
    suffix_counter = Counter()
    missing_four_ch = []
    for patient_dir in patients:
        names = {item.name for item in patient_dir.iterdir()}
        for item in patient_dir.iterdir():
            suffixes = "".join(item.suffixes) or item.suffix
            suffix_counter[suffixes] += 1
        expected = {
            f"{patient_dir.name}_4CH_ED.nii.gz",
            f"{patient_dir.name}_4CH_ED_gt.nii.gz",
            f"{patient_dir.name}_4CH_ES.nii.gz",
            f"{patient_dir.name}_4CH_ES_gt.nii.gz",
            f"{patient_dir.name}_4CH_half_sequence.nii.gz",
            f"{patient_dir.name}_4CH_half_sequence_gt.nii.gz",
            "Info_4CH.cfg",
        }
        if not expected.issubset(names):
            missing_four_ch.append(patient_dir.name)

    split_counts = {}
    split_examples = {}
    split_map = {
        "train_data": split_root / "subgroup_training.txt",
        "val_data": split_root / "subgroup_validation.txt",
        "test_data": split_root / "subgroup_testing.txt",
    }
    for key, path in split_map.items():
        items = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        split_counts[key] = len(items)
        split_examples[key] = items[:3]

    summary = {
        "source_root": str(camus_root),
        "database_root": str(db_root),
        "patient_count": len(patients),
        "sample_patients": sample_patients,
        "sample_files": sample_files,
        "file_suffix_counts": dict(sorted(suffix_counter.items())),
        "has_official_split": split_root.exists(),
        "official_split_counts": split_counts,
        "official_split_examples": split_examples,
        "missing_4ch_cases": missing_four_ch[:20],
        "notes": [
            "Each patient directory contains both 2CH and 4CH variants.",
            "ED/ES metadata is stored in Info_2CH.cfg and Info_4CH.cfg.",
            "Sequence volumes are stored as *_half_sequence.nii.gz and masks as *_half_sequence_gt.nii.gz.",
        ],
    }
    return summary


def load_volume(sitk, path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 2:
        array = array[None, ...]
    return array


def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros(volume.shape, dtype=np.uint8)
    scaled = (volume - vmin) / (vmax - vmin)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype(np.uint8), mode="L")
    resized = pil_mask.resize((size, size), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)


def choose_lv_label(mask_volume: np.ndarray, configured_label: int) -> tuple[int, str]:
    values = sorted(int(v) for v in np.unique(mask_volume) if int(v) > 0)
    if not values:
        return configured_label, "no positive label found"
    if configured_label in values:
        return configured_label, "configured label present"
    if values == [1]:
        return 1, "binary mask detected"
    if len(values) == 1:
        return values[0], "single positive label detected"
    if 1 in values:
        return 1, "fell back to CAMUS LV convention label=1"
    return values[0], "fell back to smallest positive label"


def sample_frame_indices(start_idx: int, end_idx: int, num_frames: int) -> tuple[list[int], bool]:
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    indices = np.linspace(start_idx, end_idx, num_frames)
    indices = np.clip(np.round(indices).astype(int), start_idx, end_idx).tolist()
    return indices, len(set(indices)) < num_frames


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def read_official_split(camus_root: Path) -> dict[str, list[str]]:
    split_root = camus_root / "database_split"
    mapping = {
        "train_data": split_root / "subgroup_training.txt",
        "val_data": split_root / "subgroup_validation.txt",
        "test_data": split_root / "subgroup_testing.txt",
    }
    result = {}
    for key, path in mapping.items():
        result[key] = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return result


def fallback_split(patient_ids: list[str], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = patient_ids[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.1)
    return {
        "train_data": sorted(shuffled[:train_end]),
        "val_data": sorted(shuffled[train_end:val_end]),
        "test_data": sorted(shuffled[val_end:]),
    }


def preprocess_dataset(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    sitk = ensure_simpleitk()
    configured_lv_label = getattr(args, "lv_label", 1)

    camus_root = (args.input_root / "CAMUS_public").expanduser()
    db_root = camus_root / "database_nifti"
    if not db_root.exists():
        raise FileNotFoundError(f"CAMUS root not found: {db_root}")

    output_dir = (args.output_root / DATASET_NAME).expanduser()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "preprocess.log")
    summary = scan_camus_structure(camus_root)
    LOGGER.info("CAMUS directory summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))

    patient_dirs = sorted(p for p in db_root.iterdir() if p.is_dir() and p.name.startswith("patient"))
    stats = {
        "raw_total": len(patient_dirs),
        "processed": 0,
        "skipped": 0,
        "split_counts": Counter(),
    }
    skip_reasons: Counter[str] = Counter()
    bad_cases: list[dict[str, object]] = []
    processed_patients: set[str] = set()

    for patient_dir in tqdm(patient_dirs, desc="CAMUS patients"):
        patient_id = patient_dir.name
        try:
            cfg = read_cfg(patient_dir / "Info_4CH.cfg")
            ed_idx = max(int(cfg["ED"]) - 1, 0)
            es_idx = max(int(cfg["ES"]) - 1, 0)
            half_seq = load_volume(sitk, patient_dir / f"{patient_id}_4CH_half_sequence.nii.gz")
            half_seq_gt = load_volume(sitk, patient_dir / f"{patient_id}_4CH_half_sequence_gt.nii.gz")

            if half_seq.shape != half_seq_gt.shape:
                raise ValueError(
                    f"image/mask shape mismatch: {half_seq.shape} vs {half_seq_gt.shape}"
                )

            num_available = half_seq.shape[0]
            start_idx = min(ed_idx, es_idx, num_available - 1)
            end_idx = min(max(ed_idx, es_idx), num_available - 1)
            indices, used_repeat = sample_frame_indices(start_idx, end_idx, args.num_frames)
            case_lv_label, lv_reason = choose_lv_label(half_seq_gt, configured_lv_label)

            image_u8 = normalize_to_uint8(half_seq)
            img_dir = output_dir / "img" / patient_id
            mask_dir = output_dir / "gt_lv" / patient_id
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            for out_idx, src_idx in enumerate(indices):
                img = resize_image(image_u8[src_idx], 256)
                mask = (half_seq_gt[src_idx] == case_lv_label).astype(np.uint8)
                mask = resize_mask(mask, 256)

                save_png(img, img_dir / f"{out_idx:04d}.png")
                save_png(mask, mask_dir / f"{out_idx:04d}.png")

            processed_patients.add(patient_id)
            stats["processed"] += 1
            if used_repeat:
                LOGGER.warning(
                    "CAMUS repeated frames for %s: source range [%d, %d], sampled=%s",
                    patient_id,
                    start_idx,
                    end_idx,
                    indices,
                )
            LOGGER.info(
                "CAMUS processed %s | source_frames=%d | ed=%d | es=%d | lv_label=%d (%s)",
                patient_id,
                num_available,
                ed_idx,
                es_idx,
                case_lv_label,
                lv_reason,
            )
        except Exception as exc:  # noqa: BLE001
            stats["skipped"] += 1
            reason = str(exc)
            skip_reasons[reason] += 1
            bad_cases.append({"patient_id": patient_id, "reason": reason})
            LOGGER.exception("Failed to process CAMUS case %s", patient_id)

    official_split_path = camus_root / "database_split"
    if official_split_path.exists():
        split_dict = read_official_split(camus_root)
    else:
        split_dict = fallback_split(sorted(processed_patients), args.seed)

    filtered_split = {
        key: [pid for pid in values if pid in processed_patients]
        for key, values in split_dict.items()
    }
    for key, values in filtered_split.items():
        stats["split_counts"][key] = len(values)

    (output_dir / SPLIT_JSON_NAME).write_text(
        json.dumps(filtered_split, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "camus_bad_cases.json").write_text(
        json.dumps(bad_cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "scan_summary": summary,
        "raw_total": stats["raw_total"],
        "processed": stats["processed"],
        "skipped": stats["skipped"],
        "skip_reason_counts": dict(skip_reasons),
        "split_counts": dict(stats["split_counts"]),
    }
    LOGGER.info("CAMUS preprocess summary:\n%s", json.dumps(report, indent=2, ensure_ascii=False))
    return output_dir


def main() -> None:
    args = parse_args()
    preprocess_dataset(args)


if __name__ == "__main__":
    main()
