#!/usr/bin/env python3
"""Preprocess pediatric EchoNet-style data into the weakly supervised GDKVM format."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    from tools.echonet_sampling import build_sample_plan
except ModuleNotFoundError:
    from echonet_sampling import build_sample_plan

LOGGER = logging.getLogger("preprocess_echonet_pediatric")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess pediatric echo AVI data into the format required by GDKVM."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("~/datasets/echonetpediatric").expanduser(),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("~/datasets/processed").expanduser(),
    )
    parser.add_argument("--view", type=str, default="A4C", choices=["A4C", "PSAX"])
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_visualizations", type=int, default=16)
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="ed_to_es",
        choices=["ed_to_es", "full_cycle"],
    )
    parser.add_argument("--train_folds", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--val_folds", type=str, default="8")
    parser.add_argument("--test_folds", type=str, default="9")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required to read AVI videos. Install it in the current environment."
        ) from exc
    return cv2


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


def parse_fold_spec(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def resolve_view_root(input_root: Path, view: str) -> Path:
    input_root = input_root.expanduser()
    candidates = [
        input_root / "pediatric_echo_avi" / "pediatric_echo_avi" / view,
        input_root / view,
    ]
    for candidate in candidates:
        if (candidate / "FileList.csv").exists():
            return candidate
    raise FileNotFoundError(f"Could not find pediatric view root for {view} under {input_root}")


def split_from_fold(fold: str, train_folds: set[str], val_folds: set[str], test_folds: set[str]) -> str:
    if fold in train_folds:
        return "train"
    if fold in val_folds:
        return "val"
    if fold in test_folds:
        return "test"
    raise ValueError(f"Unmapped fold value: {fold}")


def deduplicate_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def rasterize_polygon(points: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon(points, outline=1, fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def polygon_signed_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def analyze_polygon(points: list[tuple[float, float]], width: int, height: int) -> list[str]:
    warnings: list[str] = []
    if len(points) < 3:
        warnings.append("polygon has fewer than 3 vertices after deduplication")
        return warnings
    area = abs(polygon_signed_area(points))
    if area < max(width, height):
        warnings.append(f"polygon area is suspiciously small: {area:.2f}")
    out_of_bounds = [
        point for point in points
        if point[0] < 0 or point[0] >= width or point[1] < 0 or point[1] >= height
    ]
    if out_of_bounds:
        warnings.append(f"polygon has {len(out_of_bounds)} out-of-bounds points")
    return warnings


def tracing_to_mask(
    tracing_rows: list[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[np.ndarray, list[tuple[float, float]], list[str]]:
    polygon = deduplicate_points(tracing_rows)
    if len(polygon) < 3:
        raise ValueError("tracing polygon has fewer than 3 points")
    warnings = analyze_polygon(polygon, width=width, height=height)
    mask = rasterize_polygon(polygon, width=width, height=height)
    if mask.max() == 0:
        raise ValueError("tracing mask is empty after rasterization")
    return mask, polygon, warnings


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((size, size), resample=Image.Resampling.BILINEAR), dtype=np.uint8)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype(np.uint8), mode="L")
    resized = pil_mask.resize((size, size), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_overlay_visualization(
    image: np.ndarray,
    polygon: list[tuple[float, float]],
    mask: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.fromarray(image).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.polygon(polygon, outline=(255, 64, 64, 255))
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    mask_rgba = Image.new("RGBA", base.size, (80, 220, 120, 0))
    mask_rgba.putalpha(mask_img.point(lambda value: 90 if value > 0 else 0))
    overlay = Image.alpha_composite(overlay, mask_rgba)
    Image.alpha_composite(base.convert("RGBA"), overlay).save(output_path)


def read_selected_frames(cv2, video_path: Path, target_indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("failed to open video")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if min(target_indices) < 0 or max(target_indices) >= frame_count:
        raise ValueError(
            f"frame index out of range: targets={target_indices}, frame_count={frame_count}"
        )
    pending = set(target_indices)
    results: dict[int, np.ndarray] = {}
    current_idx = 0
    while pending:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in pending:
            results[current_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pending.remove(current_idx)
        current_idx += 1
    cap.release()
    if pending:
        raise ValueError(f"failed to read frames: missing {sorted(pending)}")
    return results


def read_filelist(filelist_path: Path) -> list[dict[str, str]]:
    with filelist_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_tracings(tracing_path: Path) -> dict[str, dict[int, list[tuple[float, float]]]]:
    tracings: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    with tracing_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame = row["Frame"].strip()
            if frame in {"", "No Systolic", "No Diastolic"}:
                continue
            case_name = Path(row["FileName"]).stem
            frame_idx = int(float(frame))
            tracings[case_name][frame_idx].append((float(row["X"]), float(row["Y"])))
    return tracings


def resolve_output_dataset_name(args: argparse.Namespace) -> str:
    """Return the processed dataset directory name for the pediatric task protocol."""
    suffix = f"echonet_pediatric_{args.view.lower()}"
    if args.sampling_mode == "full_cycle":
        suffix = f"{suffix}_{args.sampling_mode}"
    return f"{suffix}_png{args.image_size}_{args.num_frames}f"


def resolve_protocol_name(args: argparse.Namespace) -> str:
    if args.sampling_mode == "full_cycle":
        return "echonet_pediatric_fullcycle_sparse"
    return f"echonet_pediatric_{args.view.lower()}_endpoint"


def preprocess_dataset(args: argparse.Namespace) -> Path:
    cv2 = ensure_cv2()
    train_folds = parse_fold_spec(args.train_folds)
    val_folds = parse_fold_spec(args.val_folds)
    test_folds = parse_fold_spec(args.test_folds)
    view_root = resolve_view_root(args.input_root, args.view)
    filelist_path = view_root / "FileList.csv"
    tracing_path = view_root / "VolumeTracings.csv"
    videos_dir = view_root / "Videos"

    dataset_name = resolve_output_dataset_name(args)
    output_dir = (args.output_root / dataset_name).expanduser()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "preprocess.log")

    file_rows = read_filelist(filelist_path)
    tracings = read_tracings(tracing_path)
    summary = {
        "source_root": str(view_root),
        "video_dir": str(videos_dir),
        "filelist_rows": len(file_rows),
        "tracing_case_count": len(tracings),
        "view": args.view,
        "split_mapping": {
            "train": sorted(train_folds),
            "val": sorted(val_folds),
            "test": sorted(test_folds),
        },
    }
    LOGGER.info("Pediatric echo directory summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))

    stats = {"processed": 0, "skipped": 0, "split_counts": Counter()}
    skip_reasons: Counter[str] = Counter()
    bad_cases: list[dict[str, object]] = []
    visualized_cases = 0

    for row in tqdm(file_rows, desc=f"Pediatric {args.view} cases"):
        case_name = Path(row["FileName"].strip()).stem
        fold = row["Split"].strip()
        video_path = videos_dir / f"{case_name}.avi"
        try:
            split = split_from_fold(fold, train_folds, val_folds, test_folds)
            if case_name not in tracings:
                raise ValueError("missing tracing")
            if not video_path.exists():
                raise ValueError("missing video")

            frame_map = tracings[case_name]
            if len(frame_map) != 2:
                raise ValueError(f"expected 2 traced frames, found {len(frame_map)}")

            annotated_frames = sorted(frame_map)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError("failed to open video")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            sample_plan = build_sample_plan(
                annotated_frames,
                frame_count=frame_count,
                num_frames=args.num_frames,
                mode=args.sampling_mode,
            )
            sampled_indices = sample_plan.indices
            used_repeat = sample_plan.used_repeat
            selected_frames = read_selected_frames(cv2, video_path, sampled_indices)
            sample_frame = selected_frames[sampled_indices[0]]
            frame_height, frame_width = sample_frame.shape

            label_first_raw, polygon_first, first_warnings = tracing_to_mask(
                frame_map[annotated_frames[0]], width=frame_width, height=frame_height
            )
            label_last_raw, polygon_last, last_warnings = tracing_to_mask(
                frame_map[annotated_frames[1]], width=frame_width, height=frame_height
            )
            label_first = resize_mask(label_first_raw, args.image_size)
            label_last = resize_mask(label_last_raw, args.image_size)
            if label_first.max() == 0 or label_last.max() == 0:
                raise ValueError("empty mask after resize")

            for warning in first_warnings:
                LOGGER.warning("Polygon warning for %s frame %d: %s", case_name, annotated_frames[0], warning)
            for warning in last_warnings:
                LOGGER.warning("Polygon warning for %s frame %d: %s", case_name, annotated_frames[1], warning)

            case_img_dir = output_dir / split / "img" / case_name
            case_label_dir = output_dir / split / "label" / case_name
            case_img_dir.mkdir(parents=True, exist_ok=True)
            case_label_dir.mkdir(parents=True, exist_ok=True)

            for out_idx, src_idx in enumerate(sampled_indices):
                save_png(resize_image(selected_frames[src_idx], args.image_size), case_img_dir / f"{out_idx:04d}.png")
            save_png(label_first, case_label_dir / f"{sample_plan.label_indices[0]:04d}.png")
            save_png(label_last, case_label_dir / f"{sample_plan.label_indices[-1]:04d}.png")

            sample_meta = {
                "dataset": "echonet_pediatric",
                "mode": args.sampling_mode,
                "protocol_name": resolve_protocol_name(args),
                "num_frames": args.num_frames,
                "label_indices": sample_plan.label_indices,
                "source_frames": sampled_indices,
                "window_start": sample_plan.window_start,
                "window_end": sample_plan.window_end,
                "original_size": [frame_height, frame_width],
                "resized_size": [args.image_size, args.image_size],
                "case_name": case_name,
                "split": split,
                "view": args.view,
            }
            meta_dir = output_dir / split / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / f"{case_name}.json").write_text(
                json.dumps(sample_meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if visualized_cases < args.num_visualizations:
                first_sampled_frame = sampled_indices[sample_plan.label_indices[0]]
                last_sampled_frame = sampled_indices[sample_plan.label_indices[-1]]
                save_overlay_visualization(
                    selected_frames[first_sampled_frame],
                    polygon_first,
                    label_first_raw,
                    output_dir / "qa_overlays" / split / f"{case_name}_{sample_plan.label_indices[0]:04d}_overlay.png",
                )
                save_overlay_visualization(
                    selected_frames[last_sampled_frame],
                    polygon_last,
                    label_last_raw,
                    output_dir / "qa_overlays" / split / f"{case_name}_{sample_plan.label_indices[-1]:04d}_overlay.png",
                )
                visualized_cases += 1

            if used_repeat:
                LOGGER.warning(
                    "Repeated frames for %s: annotated=%s sampled=%s",
                    case_name,
                    annotated_frames,
                    sampled_indices,
                )

            stats["processed"] += 1
            stats["split_counts"][split] += 1
        except Exception as exc:  # noqa: BLE001
            stats["skipped"] += 1
            skip_reasons[str(exc)] += 1
            bad_cases.append({"case_name": case_name, "fold": fold, "reason": str(exc)})
            LOGGER.exception("Failed to process pediatric case %s", case_name)

    (output_dir / "pediatric_bad_cases.json").write_text(
        json.dumps(bad_cases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "raw_total": len(file_rows),
        "processed": stats["processed"],
        "skipped": stats["skipped"],
        "split_counts": dict(stats["split_counts"]),
        "skip_reason_counts": dict(skip_reasons),
        "sampling_mode": args.sampling_mode,
        "protocol_name": resolve_protocol_name(args),
        "task_contract": (
            "Endpoint-only weak supervision: 10 frames with labels at traced keyframes."
            if args.sampling_mode == "full_cycle"
            else "Endpoint-only weak supervision: 10 frames with labels at 0000.png and 0009.png."
        ),
    }
    LOGGER.info("Pediatric preprocess summary:\n%s", json.dumps(report, indent=2, ensure_ascii=False))
    return output_dir


def main() -> None:
    args = parse_args()
    preprocess_dataset(args)


if __name__ == "__main__":
    main()
