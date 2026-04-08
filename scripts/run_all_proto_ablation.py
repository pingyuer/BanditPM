#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = PROJECT_ROOT / "train.py"
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "proto_ablation"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
REPORT_MD = OUTPUT_ROOT / "report.md"
PAPER_SUMMARY_MD = OUTPUT_ROOT / "paper_summary.md"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolSpec:
    key: str
    dataset: str
    protocol_name: str
    config_name: str
    data_path: str
    frame_scope: str
    group: str


@dataclass(frozen=True)
class InitSpec:
    key: str
    init_mode: str
    name_suffix: str


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    method_name: str
    dataset: str
    protocol_name: str
    init_mode: str
    frame_scope: str
    group: str
    config_name: str
    data_path: str
    run_dir: Path
    method_overrides: tuple[str, ...]
    unavailable_reason: str | None = None

    def command(self) -> list[str]:
        return [
            str(PYTHON_BIN),
            str(TRAIN_PY),
            f"--config-name={self.config_name}",
            f"hydra.run.dir={self.run_dir.as_posix()}",
            "main_training.batch_size=1",
            "main_training.num_iterations=1000",
            "eval_stage.eval_interval=200",
            "save=0",
            "wandb_mode=offline",
            *self.method_overrides,
        ]


METHODS = (
    MethodSpec("original_gdr", ()),
    MethodSpec(
        "kpff",
        (
            "model.memory_core.type=none",
            "model.temporal_memory.type=none",
        ),
    ),
    MethodSpec(
        "bpm_rule",
        (
            "model.memory_core.type=bpm",
            "model.temporal_memory.type=bpm",
            "model.temporal_memory.bpm.ENABLE=true",
            "model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true",
            "model.temporal_memory.bpm.USE_LEARNED_POLICY=false",
            "model.temporal_memory.bpm.EXEC_POLICY=rule",
            "model.temporal_memory.bpm.ENABLE_POLICY_LOSS=false",
            "model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=false",
            "model.temporal_memory.bpm.ENABLE_RL_LOSS=false",
        ),
    ),
    MethodSpec(
        "bpm_rl",
        (
            "model.memory_core.type=bpm",
            "model.temporal_memory.type=bpm",
            "model.temporal_memory.bpm.ENABLE=true",
            "model.temporal_memory.bpm.USE_RULE_BASED_POLICY=true",
            "model.temporal_memory.bpm.USE_LEARNED_POLICY=true",
            "model.temporal_memory.bpm.EXEC_POLICY=mixed",
            "model.temporal_memory.bpm.ENABLE_POLICY_LOSS=true",
            "model.temporal_memory.bpm.ENABLE_POLICY_CE_LOSS=true",
            "model.temporal_memory.bpm.ENABLE_RL_LOSS=true",
        ),
    ),
)

PROTOCOLS = (
    ProtocolSpec(
        key="echonet_fullcycle",
        dataset="echonet",
        protocol_name="echonet_fullcycle_sparse",
        config_name="echonet_fullcycle_predinit.yaml",
        data_path="/home/tahara/datasets/processed/echonet_full_cycle_png128_10f",
        frame_scope="all_available",
        group="main",
    ),
    ProtocolSpec(
        key="echonet_ed2es_endpoint",
        dataset="echonet",
        protocol_name="echonet_ed2es_endpoint",
        config_name="echonet_ed2es_endpoint_predinit.yaml",
        data_path="/home/tahara/datasets/processed/echonet_png128_10f",
        frame_scope="supervised_only",
        group="appendix",
    ),
    ProtocolSpec(
        key="camus_short_dense",
        dataset="camus",
        protocol_name="camus_short_dense",
        config_name="camus_short_dense_predinit.yaml",
        data_path="/home/tahara/datasets/processed/camus_png256_10f",
        frame_scope="all_available",
        group="main",
    ),
    ProtocolSpec(
        key="camus_full_dense",
        dataset="camus",
        protocol_name="camus_full_dense",
        config_name="camus_full_dense_predinit.yaml",
        data_path="/home/tahara/datasets/processed/camus_full_png256_10f",
        frame_scope="all_available",
        group="appendix",
    ),
)

INITS = (
    InitSpec("predinit", "pred_or_zero", "predinit"),
    InitSpec("oracle", "oracle_gt", "oracle"),
)


def build_registry() -> list[ExperimentSpec]:
    registry: list[ExperimentSpec] = []
    for protocol in PROTOCOLS:
        for init in INITS:
            for method in METHODS:
                exp_name = f"{protocol.key}_{method.name}_{init.name_suffix}"
                unavailable_reason = None
                if not Path(protocol.data_path).exists():
                    unavailable_reason = f"missing dataset root: {protocol.data_path}"
                method_overrides = (
                    f"exp_id={exp_name}",
                    f"dataset_name={protocol.dataset}",
                    f"data_path={protocol.data_path}",
                    f"data.protocol_name={protocol.protocol_name}",
                    f"phase_init.val={init.init_mode}",
                    f"phase_init.test={init.init_mode}",
                    f"evaluation.init_mode={init.init_mode}",
                    f"evaluation.frame_scope={protocol.frame_scope}",
                    *method.overrides,
                )
                registry.append(
                    ExperimentSpec(
                        name=exp_name,
                        method_name=method.name,
                        dataset=protocol.dataset,
                        protocol_name=protocol.protocol_name,
                        init_mode=init.init_mode,
                        frame_scope=protocol.frame_scope,
                        group=protocol.group if init.key == "predinit" else "appendix",
                        config_name=protocol.config_name,
                        data_path=protocol.data_path,
                        run_dir=OUTPUT_ROOT / protocol.dataset / exp_name,
                        method_overrides=method_overrides,
                        unavailable_reason=unavailable_reason,
                    )
                )
    return registry


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return proc.wait()


def load_run_summary(run_dir: Path) -> dict[str, str] | None:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return None
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [row for row in rows if row.get("mode") == "test"]
    if test_rows:
        return test_rows[-1]
    return rows[-1] if rows else None


def write_outputs(rows: list[dict[str, str]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name",
        "method_name",
        "dataset",
        "protocol_name",
        "init_mode",
        "frame_scope",
        "metric_space",
        "dice_frame_mean",
        "dice_video_mean",
        "iou_frame_mean",
        "iou_video_mean",
        "hd95_resized",
        "hd95_original",
        "assd_resized",
        "assd_original",
        "temporal_drift",
        "best_ckpt_rule",
        "seed",
        "commit_hash",
        "status",
        "group",
        "config_name",
        "run_dir",
        "notes",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    main_rows = [row for row in rows if row.get("group") == "main" and row.get("status") == "completed"]
    appendix_rows = [row for row in rows if row.get("group") != "main"]

    report_lines = [
        "# Strict Method Validation Report",
        "",
        f"- Total experiments: `{len(rows)}`",
        f"- Main-table rows: `{len(main_rows)}`",
        "",
        "## Main Table",
        "",
        "| Experiment | Method | Dataset | Protocol | Init | Dice(video) | IoU(video) | HD95(orig) | ASSD(orig) | Drift | Status |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in main_rows:
        report_lines.append(
            f"| {row['experiment_name']} | {row['method_name']} | {row['dataset']} | {row['protocol_name']} | {row['init_mode']} | "
            f"{row['dice_video_mean']} | {row['iou_video_mean']} | {row['hd95_original']} | {row['assd_original']} | {row['temporal_drift']} | {row['status']} |"
        )

    report_lines.extend(
        [
            "",
            "## Appendix",
            "",
            "| Experiment | Method | Protocol | Init | Status | Notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in appendix_rows:
        report_lines.append(
            f"| {row['experiment_name']} | {row.get('method_name', '')} | {row['protocol_name']} | {row['init_mode']} | {row['status']} | {row.get('notes', '')} |"
        )
    REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    paper_lines = [
        "# Paper Summary",
        "",
        "Main results default to stricter protocols with pred-init and include method comparisons.",
        "",
        "## Main Results",
        "",
    ]
    for row in main_rows:
        paper_lines.append(
            f"- `{row['experiment_name']}`: method=`{row['method_name']}`, protocol=`{row['protocol_name']}`, "
            f"init=`{row['init_mode']}`, dice_video_mean=`{row['dice_video_mean']}`, "
            f"iou_video_mean=`{row['iou_video_mean']}`, hd95_original=`{row['hd95_original']}`, "
            f"assd_original=`{row['assd_original']}`, temporal_drift=`{row['temporal_drift']}`"
        )

    paper_lines.extend(
        [
            "",
            "## Appendix Results",
            "",
        ]
    )
    for row in appendix_rows:
        paper_lines.append(
            f"- `{row['experiment_name']}`: method=`{row.get('method_name', '')}`, protocol=`{row['protocol_name']}`, "
            f"init=`{row['init_mode']}`, status=`{row['status']}`"
        )
    PAPER_SUMMARY_MD.write_text("\n".join(paper_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict protocol and method validation experiments.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    registry = build_registry()

    for spec in registry:
        base_row = {
            "experiment_name": spec.name,
            "method_name": spec.method_name,
            "dataset": spec.dataset,
            "protocol_name": spec.protocol_name,
            "init_mode": spec.init_mode,
            "frame_scope": spec.frame_scope,
            "metric_space": "original",
            "best_ckpt_rule": "max_eval_dice_observed_no_reload",
            "group": spec.group,
            "config_name": spec.config_name,
            "run_dir": spec.run_dir.as_posix(),
            "status": "pending",
            "notes": "",
        }
        if spec.unavailable_reason is not None:
            base_row["status"] = "unavailable"
            base_row["notes"] = spec.unavailable_reason
            rows.append(base_row)
            continue

        command = spec.command()
        if args.dry_run:
            base_row["status"] = "dry_run"
            base_row["notes"] = shlex.join(command)
            rows.append(base_row)
            continue

        log_path = spec.run_dir / "launcher.log"
        return_code = run_command(command, log_path)
        if return_code != 0:
            base_row["status"] = "failed"
            base_row["notes"] = f"exit_code={return_code}"
            rows.append(base_row)
            continue

        summary_row = load_run_summary(spec.run_dir)
        if summary_row is None:
            base_row["status"] = "failed"
            base_row["notes"] = "missing per-run summary.csv"
            rows.append(base_row)
            continue

        summary_row["method_name"] = spec.method_name
        summary_row["status"] = "completed"
        summary_row["group"] = spec.group
        summary_row["config_name"] = spec.config_name
        summary_row["run_dir"] = spec.run_dir.as_posix()
        summary_row.setdefault("notes", "")
        rows.append(summary_row)

    write_outputs(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
