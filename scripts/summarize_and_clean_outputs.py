from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


FIELDS = [
    "experiment_name",
    "method",
    "dataset",
    "protocol",
    "version",
    "latest_iter",
    "latest_test_dice",
    "best_test_dice",
    "best_iter",
    "iou",
    "hd95",
    "asd",
    "temporal_dice",
    "best_threshold",
    "source_summary_path",
]


def _float_or_none(value: str | None) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _get(row: dict, *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _infer_method(name: str) -> str:
    lower = name.lower()
    if "unext" in lower or "midfusion" in lower or "memory_primary" in lower or "spatial_phase" in lower:
        return "unext_fusion"
    if "kpff" in lower:
        return "kpff"
    if "gdkvm" in lower or "banditpm" in lower or "bpm" in lower or "dynakey" in lower:
        return "gdkvm"
    return "unknown"


def _summarize_file(path: Path, root: Path) -> dict | None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [row for row in rows if str(row.get("mode", "")).lower() == "test"]
    if not test_rows:
        return None

    latest = test_rows[-1]
    best = max(
        test_rows,
        key=lambda row: _float_or_none(_get(row, "dice_frame_mean", "Test Dice", "DICE_FRAME_MEAN")) or -1.0,
    )
    experiment = _get(latest, "experiment_name", "Name") or path.relative_to(root).parts[0]
    dataset = _get(latest, "dataset", "Dataset")
    return {
        "experiment_name": experiment,
        "method": _infer_method(experiment),
        "dataset": dataset,
        "protocol": _get(latest, "protocol_name"),
        "version": _get(latest, "protocol_version"),
        "latest_iter": _get(latest, "iteration", "Itr"),
        "latest_test_dice": _get(latest, "dice_frame_mean", "Test Dice", "DICE_FRAME_MEAN"),
        "best_test_dice": _get(best, "dice_frame_mean", "Test Dice", "DICE_FRAME_MEAN"),
        "best_iter": _get(best, "iteration", "Itr"),
        "iou": _get(latest, "iou_frame_mean", "Test Jaccard", "IOU_FRAME_MEAN"),
        "hd95": _get(latest, "hd95_original", "hd95_resized", "Test HD95"),
        "asd": _get(latest, "assd_original", "assd_resized", "Test ASD"),
        "temporal_dice": _get(latest, "temporal_dice_consistency", "Test Temporal Dice Consistency"),
        "best_threshold": _get(latest, "best_val_threshold", "Best Val Threshold"),
        "source_summary_path": str(path),
    }


def summarize(root: Path, output: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("summary.csv")):
        if path.resolve() == output.resolve():
            continue
        summary = _summarize_file(path, root)
        if summary is not None:
            rows.append(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def clean_outputs(root: Path, keep: Path) -> None:
    root = root.resolve()
    keep = keep.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Output root does not exist or is not a directory: {root}")
    if keep.parent.resolve() != root:
        raise ValueError(f"Summary file must live directly under output root: keep={keep}, root={root}")
    if root in {Path("/").resolve(), Path.cwd().resolve()}:
        raise ValueError(f"Refusing to clean unsafe root: {root}")
    for child in root.iterdir():
        if child.resolve() == keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize outputs and optionally clean historical runs.")
    parser.add_argument("--root", default="outputs", type=Path)
    parser.add_argument("--output", default=Path("outputs") / "EXPERIMENT_SUMMARY.csv", type=Path)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rows = summarize(args.root, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.clean:
        clean_outputs(args.root, args.output)
        print(f"Cleaned {args.root}; kept {args.output}")


if __name__ == "__main__":
    main()
