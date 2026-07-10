from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


IMPORTANT_PATHS = (
    "config.exp_id",
    "config.seed",
    "config.model.name",
    "config.model.backbone_pretrained",
    "config.model.pretrain_source",
    "config.temporal_access",
    "config.prediction_mode",
    "config.main_training.num_iterations",
    "config.main_training.batch_size",
    "config.main_training.amp",
    "config.main_training.learning_rate",
    "config.main_training.lr_schedule",
    "data_flow.dataset",
    "data_flow.data_path",
    "data_flow.seq_length",
    "data_flow.splits.train",
    "data_flow.splits.val",
    "data_flow.splits.test",
    "data_flow.supervision.supervised_frames_mean",
    "data_flow.supervision.label_sparsity",
    "data_flow.protocol.effective_batch_size",
    "data_flow.protocol.world_size",
    "data_flow.protocol.backbone_pretrained",
    "data_flow.protocol.pretrain_source",
    "runtime.torch",
    "runtime.cuda",
    "runtime.cudnn",
    "runtime.cudnn_benchmark",
    "runtime.cudnn_deterministic",
    "git.commit",
    "git.dirty",
    "summary.dice_frame_mean",
    "summary.iou_frame_mean",
    "summary.best_val_threshold",
    "summary.dpfr/anchor_dice",
    "summary.dpfr/prompt_dice",
    "summary.dpfr/flow_dice",
    "summary.dpfr/final_minus_anchor_dice",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"__missing__": True}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__error__": str(exc)}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"__missing__": True}
    try:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    except Exception as exc:
        return {"__error__": str(exc)}


def load_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    return {
        "run_dir": str(root),
        "config": _read_yaml(root / "config_resolved.yaml"),
        "runtime": _read_json(root / "runtime.json"),
        "git": _read_json(root / "git.json"),
        "data_flow": _read_json(root / "data_flow_summary.json"),
        "summary": _read_json(root / "summary.json"),
        "best": _read_json(root / "best_summary.json"),
    }


def _get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return "unknown"
    return current


def compare_runs(left_dir: str | Path, right_dir: str | Path) -> dict[str, Any]:
    left = load_run_artifacts(left_dir)
    right = load_run_artifacts(right_dir)
    rows = []
    for path in IMPORTANT_PATHS:
        left_value = _get(left, path)
        right_value = _get(right, path)
        rows.append(
            {
                "field": path,
                "left": left_value,
                "right": right_value,
                "match": left_value == right_value,
            }
        )
    return {
        "left_run_dir": str(left_dir),
        "right_run_dir": str(right_dir),
        "differences": [row for row in rows if not row["match"]],
        "all_fields": rows,
        "interpretation": diagnostic_interpretation(rows),
    }


def diagnostic_interpretation(rows: list[dict[str, Any]]) -> list[str]:
    by_field = {row["field"]: row for row in rows}
    notes = []
    for field in (
        "git.commit",
        "config.seed",
        "data_flow.data_path",
        "data_flow.splits.train",
        "data_flow.protocol.effective_batch_size",
        "config.main_training.amp",
        "runtime.torch",
        "runtime.cuda",
        "data_flow.protocol.backbone_pretrained",
        "summary.best_val_threshold",
    ):
        row = by_field.get(field)
        if row and not row["match"]:
            notes.append(f"{field} differs; this can invalidate a direct score comparison.")
    anchor = by_field.get("summary.dpfr/anchor_dice")
    gain = by_field.get("summary.dpfr/final_minus_anchor_dice")
    if anchor and anchor["left"] != "unknown" and anchor["right"] != "unknown":
        notes.append("DPFR anchor dice is available; compare it before blaming prompt/flow refinement.")
    if gain and gain["left"] != "unknown" and gain["right"] != "unknown":
        notes.append("DPFR final-minus-anchor is available; negative values suggest refinement hurts the task.")
    if not notes:
        notes.append("No decisive artifact difference found in the standard fields; inspect raw metrics and visual panels next.")
    return notes


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Run Comparison",
        "",
        f"- left: `{report['left_run_dir']}`",
        f"- right: `{report['right_run_dir']}`",
        "",
        "## Differences",
        "",
        "| field | left | right |",
        "| --- | --- | --- |",
    ]
    for row in report["differences"]:
        lines.append(f"| `{row['field']}` | `{row['left']}` | `{row['right']}` |")
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {note}" for note in report["interpretation"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two GDKVM/DPFR run artifact directories.")
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args(argv)
    report = compare_runs(args.left, args.right)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(format_markdown(report))


if __name__ == "__main__":
    main()
