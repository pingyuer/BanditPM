#!/usr/bin/env python3
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


LOGGER = logging.getLogger("preprocess_echonet")
DATASET_NAME = "echonet_png128_10f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess EchoNet-Dynamic into the format required by GDKVM."
    )
    parser.add_argument("--input_root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("~/datasets/processed").expanduser(),
    )
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_visualizations", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required to read EchoNet videos. "
            "Install it with: pip install opencv-python"
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


def detect_split_column(fieldnames: list[str]) -> str:
    candidates = [name for name in fieldnames if name.strip().lower() == "split"]
    if candidates:
        return candidates[0]
    fuzzy = [name for name in fieldnames if "split" in name.strip().lower()]
    if fuzzy:
        return fuzzy[0]
    raise ValueError(f"Could not identify split column from headers: {fieldnames}")


def scan_echonet_structure(echonet_root: Path) -> dict[str, object]:
    filelist_path = echonet_root / "FileList.csv"
    tracing_path = echonet_root / "VolumeTracings.csv"
    videos_dir = echonet_root / "Videos"

    with filelist_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        filelist_rows = list(reader)
        split_column = detect_split_column(reader.fieldnames or [])

    split_counts = Counter(row[split_column].strip().upper() for row in filelist_rows)
    video_files = sorted(videos_dir.glob("*.avi"))
    video_stems = {path.stem for path in video_files}

    tracing_cases: dict[str, set[int]] = defaultdict(set)
    with tracing_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_name = Path(row["FileName"]).stem
            tracing_cases[case_name].add(int(float(row["Frame"])))

    summary = {
        "source_root": str(echonet_root),
        "video_dir": str(videos_dir),
        "video_format": ".avi",
        "filelist_path": str(filelist_path),
        "tracing_path": str(tracing_path),
        "filelist_rows": len(filelist_rows),
        "video_count": len(video_files),
        "tracing_case_count": len(tracing_cases),
        "split_column": split_column,
        "split_counts": dict(split_counts),
        "sample_video_names": [path.name for path in video_files[:5]],
        "sample_tracing_frames": {
            key: sorted(value)
            for key, value in list(tracing_cases.items())[:5]
        },
        "missing_tracing_cases": sorted(video_stems - set(tracing_cases))[:20],
        "missing_video_cases": sorted(set(tracing_cases) - video_stems)[:20],
        "notes": [
            "Official split comes from FileList.csv.",
            "VolumeTracings.csv contains two annotated frames per case.",
            "Each tracing frame contributes 21 contour point pairs (X1,Y1,X2,Y2).",
        ],
    }
    return summary


def read_filelist(filelist_path: Path) -> tuple[list[dict[str, str]], str]:
    with filelist_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        split_column = detect_split_column(reader.fieldnames or [])
    return rows, split_column


def read_tracings(tracing_path: Path) -> dict[str, dict[int, list[tuple[float, float, float, float]]]]:
    tracings: dict[str, dict[int, list[tuple[float, float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    with tracing_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_name = Path(row["FileName"]).stem
            frame_idx = int(float(row["Frame"]))
            tracings[case_name][frame_idx].append(
                (
                    float(row["X1"]),
                    float(row["Y1"]),
                    float(row["X2"]),
                    float(row["Y2"]),
                )
            )
    return tracings


def deduplicate_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def sample_frame_indices(start_idx: int, end_idx: int, num_frames: int) -> tuple[list[int], bool]:
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    indices = np.linspace(start_idx, end_idx, num_frames)
    indices = np.clip(np.round(indices).astype(int), start_idx, end_idx).tolist()
    return indices, len(set(indices)) < num_frames


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


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def _segments_intersect(
    p1: tuple[float, float],
    q1: tuple[float, float],
    p2: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    eps = 1e-6
    if abs(o1) <= eps and _on_segment(p1, p2, q1):
        return True
    if abs(o2) <= eps and _on_segment(p1, q2, q1):
        return True
    if abs(o3) <= eps and _on_segment(p2, p1, q2):
        return True
    if abs(o4) <= eps and _on_segment(p2, q1, q2):
        return True
    return False


def polygon_has_self_intersection(points: list[tuple[float, float]]) -> bool:
    num_points = len(points)
    if num_points < 4:
        return False
    for idx in range(num_points):
        edge_a = (points[idx], points[(idx + 1) % num_points])
        for jdx in range(idx + 1, num_points):
            if abs(idx - jdx) <= 1:
                continue
            if idx == 0 and jdx == num_points - 1:
                continue
            edge_b = (points[jdx], points[(jdx + 1) % num_points])
            if _segments_intersect(edge_a[0], edge_a[1], edge_b[0], edge_b[1]):
                return True
    return False


def analyze_polygon(points: list[tuple[float, float]], width: int, height: int) -> list[str]:
    warnings: list[str] = []
    if len(points) < 3:
        warnings.append("polygon has fewer than 3 vertices after deduplication")
        return warnings

    area = abs(polygon_signed_area(points))
    if area < max(width, height):
        warnings.append(f"polygon area is suspiciously small: {area:.2f}")

    if polygon_has_self_intersection(points):
        warnings.append("polygon has a potential self-intersection")

    out_of_bounds = [
        point for point in points
        if point[0] < 0 or point[0] >= width or point[1] < 0 or point[1] >= height
    ]
    if out_of_bounds:
        warnings.append(f"polygon has {len(out_of_bounds)} out-of-bounds points")

    y_values = [point[1] for point in points]
    if max(y_values) - min(y_values) < 2:
        warnings.append("polygon vertical span is extremely small")

    return warnings


def tracing_to_mask(
    tracing_rows: list[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> tuple[np.ndarray, list[tuple[float, float]], list[str]]:
    left = [(x1, y1) for x1, y1, _, _ in tracing_rows]
    right = [(x2, y2) for _, _, x2, y2 in tracing_rows]
    polygon = deduplicate_points(left + list(reversed(right)))
    if len(polygon) < 3:
        raise ValueError("tracing polygon has fewer than 3 points")
    warnings = analyze_polygon(polygon, width=width, height=height)
    mask = rasterize_polygon(polygon, width=width, height=height)
    if mask.max() == 0:
        raise ValueError("tracing mask is empty after rasterization")
    return mask, polygon, warnings


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


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

    if polygon:
        draw.polygon(polygon, outline=(255, 64, 64, 255))

    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    mask_rgba = Image.new("RGBA", base.size, (80, 220, 120, 0))
    mask_rgba.putalpha(mask_img.point(lambda value: 90 if value > 0 else 0))
    overlay = Image.alpha_composite(overlay, mask_rgba)

    preview = Image.alpha_composite(base.convert("RGBA"), overlay)
    preview.save(output_path)


def read_selected_frames(cv2, video_path: Path, target_indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("failed to open video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not target_indices:
        raise ValueError("no target frames requested")
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results[current_idx] = gray
            pending.remove(current_idx)
        current_idx += 1

    cap.release()
    if pending:
        raise ValueError(f"failed to read frames: missing {sorted(pending)}")
    return results


def preprocess_dataset(args: argparse.Namespace) -> Path:
    cv2 = ensure_cv2()

    echonet_root = (args.input_root / "EchoNet-Dynamic").expanduser()
    filelist_path = echonet_root / "FileList.csv"
    tracing_path = echonet_root / "VolumeTracings.csv"
    videos_dir = echonet_root / "Videos"
    if not (filelist_path.exists() and tracing_path.exists() and videos_dir.exists()):
        raise FileNotFoundError(f"EchoNet-Dynamic structure is incomplete under {echonet_root}")

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
    summary = scan_echonet_structure(echonet_root)
    LOGGER.info("EchoNet directory summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))

    file_rows, split_column = read_filelist(filelist_path)
    tracings = read_tracings(tracing_path)

    stats = {
        "raw_total": len(file_rows),
        "processed": 0,
        "skipped": 0,
        "split_counts": Counter(),
    }
    skip_reasons: Counter[str] = Counter()
    bad_cases: list[dict[str, object]] = []
    visualized_cases = 0
    qa_warning_counts: Counter[str] = Counter()

    for row in tqdm(file_rows, desc="EchoNet cases"):
        case_name = row["FileName"].strip()
        split = row[split_column].strip().lower()
        video_path = videos_dir / f"{case_name}.avi"

        try:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"unsupported split value: {row[split_column]}")
            if case_name not in tracings:
                raise ValueError("missing tracing")
            if not video_path.exists():
                raise ValueError("missing video")

            frame_map = tracings[case_name]
            if len(frame_map) != 2:
                raise ValueError(f"expected 2 traced frames, found {len(frame_map)}")

            annotated_frames = sorted(frame_map)
            sampled_indices, used_repeat = sample_frame_indices(
                annotated_frames[0], annotated_frames[1], args.num_frames
            )
            selected_frames = read_selected_frames(cv2, video_path, sampled_indices)

            frame_width = int(float(row["FrameWidth"]))
            frame_height = int(float(row["FrameHeight"]))

            label_first_raw, polygon_first, first_warnings = tracing_to_mask(
                frame_map[annotated_frames[0]],
                width=frame_width,
                height=frame_height,
            )
            label_last_raw, polygon_last, last_warnings = tracing_to_mask(
                frame_map[annotated_frames[1]],
                width=frame_width,
                height=frame_height,
            )

            label_first = resize_mask(label_first_raw, 128)
            label_last = resize_mask(label_last_raw, 128)

            if label_first.max() == 0 or label_last.max() == 0:
                raise ValueError("empty mask after resize")

            for warning in first_warnings:
                qa_warning_counts[warning] += 1
                LOGGER.warning(
                    "EchoNet polygon warning for %s frame %d: %s",
                    case_name,
                    annotated_frames[0],
                    warning,
                )
            for warning in last_warnings:
                qa_warning_counts[warning] += 1
                LOGGER.warning(
                    "EchoNet polygon warning for %s frame %d: %s",
                    case_name,
                    annotated_frames[1],
                    warning,
                )

            case_img_dir = output_dir / split / "img" / case_name
            case_label_dir = output_dir / split / "label" / case_name
            case_img_dir.mkdir(parents=True, exist_ok=True)
            case_label_dir.mkdir(parents=True, exist_ok=True)

            for out_idx, src_idx in enumerate(sampled_indices):
                img = resize_image(selected_frames[src_idx], 128)
                save_png(img, case_img_dir / f"{out_idx:04d}.png")

            save_png(label_first, case_label_dir / "0000.png")
            save_png(label_last, case_label_dir / "0009.png")

            if visualized_cases < args.num_visualizations:
                save_overlay_visualization(
                    selected_frames[annotated_frames[0]],
                    polygon_first,
                    label_first_raw,
                    output_dir / "qa_overlays" / split / f"{case_name}_0000_overlay.png",
                )
                save_overlay_visualization(
                    selected_frames[annotated_frames[1]],
                    polygon_last,
                    label_last_raw,
                    output_dir / "qa_overlays" / split / f"{case_name}_0009_overlay.png",
                )
                visualized_cases += 1

            stats["processed"] += 1
            stats["split_counts"][split] += 1
            if used_repeat:
                LOGGER.warning(
                    "EchoNet repeated frames for %s: annotated=%s sampled=%s",
                    case_name,
                    annotated_frames,
                    sampled_indices,
                )
        except Exception as exc:  # noqa: BLE001
            stats["skipped"] += 1
            reason = str(exc)
            skip_reasons[reason] += 1
            bad_cases.append(
                {
                    "case_name": case_name,
                    "split": row.get(split_column, ""),
                    "reason": reason,
                }
            )
            LOGGER.exception("Failed to process EchoNet case %s", case_name)

    (output_dir / "echonet_bad_cases.json").write_text(
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
        "qa_warning_counts": dict(qa_warning_counts),
        "qa_overlay_dir": str(output_dir / "qa_overlays"),
        "task_contract": "Endpoint-only weak supervision: 10 frames with labels only at 0000.png and 0009.png.",
    }
    LOGGER.info("EchoNet preprocess summary:\n%s", json.dumps(report, indent=2, ensure_ascii=False))
    return output_dir


def main() -> None:
    args = parse_args()
    preprocess_dataset(args)


if __name__ == "__main__":
    main()
