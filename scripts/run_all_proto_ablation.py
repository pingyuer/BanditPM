#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCHRUN_BIN = PROJECT_ROOT / ".venv" / "bin" / "torchrun"
TRAIN_PY = PROJECT_ROOT / "train.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "proto_ablation"
SUMMARY_CSV_DEFAULT = OUTPUT_ROOT / "summary.csv"
SUMMARY_TXT_DEFAULT = OUTPUT_ROOT / "summary.txt"
SUMMARY_MD_DEFAULT = OUTPUT_ROOT / "report.md"
WANDB_ROOT = PROJECT_ROOT / "wandb"
WANDB_SYNC_LOG_DEFAULT = OUTPUT_ROOT / "wandb_sync.log"

TOTAL_ITER_OVERRIDE = "main_training.num_iterations=2000"
EVAL_INTERVAL_OVERRIDE = "eval_stage.eval_interval=250"
SAVE_WEIGHTS_INTERVAL_OVERRIDE = "save_weights_interval=1000"
SAVE_CHECKPOINT_INTERVAL_OVERRIDE = "save_checkpoint_interval=1000"

DATASET_SPECS = {
    "camus": {
        "dataset_name": "camus",
        "data_path": "/home/tahara/datasets/processed/camus_png256_10f",
    },
    "echonet": {
        "dataset_name": "echonet",
        "data_path": "/home/tahara/datasets/processed/echonet_png128_10f",
    },
}

BASE_CONFIGS = {
    "baseline": "config_gdkvm_01.yaml",
    "proto_fast": "config_gdkvm_proto_fast.yaml",
    "proto_slow": "config_gdkvm_proto_slow.yaml",
    "proto_replace": "config_gdkvm_proto_ablate_replace.yaml",
    "proto_concat": "config_gdkvm_proto_ablate_fuse_concat.yaml",
    "proto_no_temporal": "config_gdkvm_proto_ablate_no_temporal.yaml",
}

CONFIG_HPARAM_DEFAULTS = {
    "config_gdkvm_01.yaml": {
        "prototype_enabled": False,
        "module_mode": "none",
        "proto_mode": "none",
        "num_proto": None,
        "temperature": None,
        "momentum": None,
        "fuse": None,
        "feature_source": None,
    },
    "config_gdkvm_proto_fast.yaml": {
        "prototype_enabled": True,
        "module_mode": "fast",
        "proto_mode": "augment",
        "num_proto": 16,
        "temperature": 1.0,
        "momentum": 0.9,
        "fuse": "add",
        "feature_source": "value",
    },
    "config_gdkvm_proto_slow.yaml": {
        "prototype_enabled": True,
        "module_mode": "slow",
        "proto_mode": "augment",
        "num_proto": 16,
        "temperature": 1.0,
        "momentum": 0.9,
        "fuse": "gated",
        "feature_source": "value",
    },
    "config_gdkvm_proto_ablate_replace.yaml": {
        "prototype_enabled": True,
        "module_mode": "fast",
        "proto_mode": "replace",
        "num_proto": 16,
        "temperature": 1.0,
        "momentum": 0.9,
        "fuse": "add",
        "feature_source": "value",
    },
    "config_gdkvm_proto_ablate_fuse_concat.yaml": {
        "prototype_enabled": True,
        "module_mode": "fast",
        "proto_mode": "augment",
        "num_proto": 16,
        "temperature": 1.0,
        "momentum": 0.9,
        "fuse": "concat",
        "feature_source": "value",
    },
    "config_gdkvm_proto_ablate_no_temporal.yaml": {
        "prototype_enabled": True,
        "module_mode": "fast",
        "proto_mode": "augment",
        "num_proto": 16,
        "temperature": 1.0,
        "momentum": 0.9,
        "fuse": "gated",
        "feature_source": "value",
    },
}

VAL_RE = re.compile(r"\[Val\]\s+Iter=(?P<iter>\d+)\s+\|\s+.*?DICE=(?P<dice>[0-9.]+)")
TEST_RE = re.compile(r"\[Test\]\s+Iter=(?P<iter>\d+)\s+\|\s+.*?DICE=(?P<dice>[0-9.]+)")
TOTAL_ITER_RE = re.compile(r"Total iterations:\s*(?P<iter>\d+)")
COMPLETE_RE = re.compile(r"Training completed\.")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset: str
    group: str
    config_name: str
    variant: str
    overrides: tuple[str, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def run_dir(self) -> Path:
        return OUTPUT_ROOT / self.dataset / self.group / self.name

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "experiment_spec.json"

    @property
    def complete_path(self) -> Path:
        return self.run_dir / ".complete"

    @property
    def launcher_log_path(self) -> Path:
        return self.run_dir / "launcher.log"

    def all_overrides(self) -> list[str]:
        dataset_spec = DATASET_SPECS[self.dataset]
        shared = [
            f"dataset_name={dataset_spec['dataset_name']}",
            f"data_path={dataset_spec['data_path']}",
            f"exp_id={self.name}",
            TOTAL_ITER_OVERRIDE,
            EVAL_INTERVAL_OVERRIDE,
            SAVE_WEIGHTS_INTERVAL_OVERRIDE,
            SAVE_CHECKPOINT_INTERVAL_OVERRIDE,
            f"hydra.run.dir={self.run_dir.as_posix()}",
        ]
        return shared + list(self.overrides)

    def command(self) -> list[str]:
        num_gpus = detect_num_gpus()
        return [
            str(TORCHRUN_BIN),
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            str(TRAIN_PY),
            f"--config-name={self.config_name}",
            *self.all_overrides(),
        ]

    def shell_command(self) -> str:
        return shlex.join(self.command())


def build_registry(include_encoder_feature_source: bool = False) -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []

    core_variants = [
        ("baseline", BASE_CONFIGS["baseline"], ()),
        ("proto_fast", BASE_CONFIGS["proto_fast"], ()),
        ("proto_slow", BASE_CONFIGS["proto_slow"], ()),
    ]
    ablation_variants = [
        ("proto_replace", BASE_CONFIGS["proto_replace"], ()),
        ("proto_concat", BASE_CONFIGS["proto_concat"], ()),
        ("proto_no_temporal", BASE_CONFIGS["proto_no_temporal"], ()),
    ]
    tune_variants = [
        (
            "proto_fast_num_proto_8",
            BASE_CONFIGS["proto_fast"],
            ("model.prototype_value.bank.num_proto=8",),
        ),
        (
            "proto_fast_num_proto_32",
            BASE_CONFIGS["proto_fast"],
            ("model.prototype_value.bank.num_proto=32",),
        ),
        (
            "proto_fast_temperature_0p5",
            BASE_CONFIGS["proto_fast"],
            ("model.prototype_value.bank.temperature=0.5",),
        ),
        (
            "proto_fast_temperature_2p0",
            BASE_CONFIGS["proto_fast"],
            ("model.prototype_value.bank.temperature=2.0",),
        ),
        (
            "proto_fast_fuse_gated",
            BASE_CONFIGS["proto_fast"],
            ("model.prototype_value.fuse.type=gated",),
        ),
        (
            "proto_slow_momentum_0p7",
            BASE_CONFIGS["proto_slow"],
            ("model.prototype_value.temporal.momentum=0.7",),
        ),
    ]
    if include_encoder_feature_source:
        tune_variants.append(
            (
                "proto_fast_feature_encoder",
                BASE_CONFIGS["proto_fast"],
                ("model.prototype_value.feature_source=encoder",),
            )
        )

    for dataset in DATASET_SPECS:
        for variant, config_name, overrides in core_variants:
            name = f"{dataset}_{variant}"
            experiments.append(
                ExperimentSpec(
                    name=name,
                    dataset=dataset,
                    group="core",
                    config_name=config_name,
                    variant=variant,
                    overrides=tuple(overrides),
                    tags=(dataset, "core", variant),
                )
            )

        for variant, config_name, overrides in ablation_variants:
            name = f"{dataset}_{variant}"
            experiments.append(
                ExperimentSpec(
                    name=name,
                    dataset=dataset,
                    group="ablation",
                    config_name=config_name,
                    variant=variant,
                    overrides=tuple(overrides),
                    tags=(dataset, "ablation", variant),
                )
            )

        for variant, config_name, overrides in tune_variants:
            name = f"{dataset}_{variant}"
            experiments.append(
                ExperimentSpec(
                    name=name,
                    dataset=dataset,
                    group="tune",
                    config_name=config_name,
                    variant=variant,
                    overrides=tuple(overrides),
                    tags=(dataset, "tune", variant),
                )
            )

    return experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all prototype ablation experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--run", action="store_true", help="Execute experiments sequentially.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=sorted(DATASET_SPECS),
        help="Datasets to include.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["core", "ablation", "tune"],
        default=["core", "ablation", "tune"],
        help="Experiment groups to include.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional substring filter on experiment name/variant.",
    )
    parser.add_argument("--max-jobs", type=int, default=1, help="Reserved. First version runs sequentially only.")
    parser.add_argument("--sleep-between-jobs", type=float, default=0.0, help="Sleep between runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments already marked complete.")
    parser.add_argument("--resume", action="store_true", help="Alias of --skip-existing for scheduler progress.")
    parser.add_argument("--export-txt", type=Path, default=None, help="Write commands to a text file.")
    parser.add_argument("--export-sh", type=Path, default=None, help="Write commands to a shell script.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=SUMMARY_CSV_DEFAULT,
        help="CSV path for collected metrics.",
    )
    parser.add_argument(
        "--summary-txt",
        type=Path,
        default=SUMMARY_TXT_DEFAULT,
        help="TXT path for collected metrics and key hyperparameters.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=SUMMARY_MD_DEFAULT,
        help="Markdown report path for paper-ready experiment summary.",
    )
    parser.add_argument(
        "--collect-results",
        action="store_true",
        help="Collect summary CSV for the selected experiments.",
    )
    parser.add_argument(
        "--include-feature-source-encoder",
        action="store_true",
        help="Include the optional encoder feature-source tune runs.",
    )
    parser.add_argument(
        "--sync-wandb",
        action="store_true",
        help="Sync offline W&B runs after execution or collection.",
    )
    parser.add_argument(
        "--no-sync-wandb",
        action="store_true",
        help="Do not sync W&B after --run.",
    )
    parser.add_argument(
        "--wandb-sync-log",
        type=Path,
        default=WANDB_SYNC_LOG_DEFAULT,
        help="Log path for wandb sync output.",
    )
    return parser.parse_args()


def detect_num_gpus() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").strip()
    if not visible:
        return 1
    num_gpus = len([item for item in visible.split(",") if item.strip()])
    return max(num_gpus, 1)


def build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(random.randint(20000, 59999))
    env["HYDRA_FULL_ERROR"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else f"{PROJECT_ROOT}:{existing_pythonpath}"
    return env


def select_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
    experiments = build_registry(include_encoder_feature_source=args.include_feature_source_encoder)
    selected = [
        exp
        for exp in experiments
        if exp.dataset in args.datasets and exp.group in args.groups
    ]
    if args.variants:
        needles = [needle.lower() for needle in args.variants]
        selected = [
            exp
            for exp in selected
            if any(needle in exp.name.lower() or needle in exp.variant.lower() for needle in needles)
        ]
    return selected


def write_exports(experiments: Iterable[ExperimentSpec], txt_path: Path | None, sh_path: Path | None) -> None:
    commands = [exp.shell_command() for exp in experiments]
    if txt_path is not None:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(commands) + ("\n" if commands else ""))
    if sh_path is not None:
        sh_path.parent.mkdir(parents=True, exist_ok=True)
        content = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + ("\n" if commands else "")
        sh_path.write_text(content)
        sh_path.chmod(0o755)


def read_completion_status(exp: ExperimentSpec) -> bool:
    if exp.complete_path.exists():
        return True
    if not exp.launcher_log_path.exists():
        return False
    try:
        return COMPLETE_RE.search(exp.launcher_log_path.read_text(errors="ignore")) is not None
    except OSError:
        return False


def write_metadata(exp: ExperimentSpec) -> None:
    exp.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": exp.name,
        "dataset": exp.dataset,
        "group": exp.group,
        "variant": exp.variant,
        "config_name": exp.config_name,
        "tags": list(exp.tags),
        "overrides": exp.all_overrides(),
        "command": exp.command(),
        "shell_command": exp.shell_command(),
    }
    exp.metadata_path.write_text(json.dumps(payload, indent=2) + "\n")


def stream_subprocess(cmd: list[str], log_path: Path) -> int:
    env = build_runtime_env()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            "ENV "
            f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
            f"MASTER_ADDR={env['MASTER_ADDR']} "
            f"MASTER_PORT={env['MASTER_PORT']}\n"
        )
        log_file.write(f"$ {shlex.join(cmd)}\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
            process.stdout.close()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise


def run_experiments(experiments: list[ExperimentSpec], args: argparse.Namespace) -> int:
    if args.max_jobs != 1:
        print("--max-jobs > 1 is not implemented in the first version; using sequential execution.", file=sys.stderr)

    skip_existing = args.skip_existing or args.resume
    failures = 0
    for index, exp in enumerate(experiments, start=1):
        if skip_existing and read_completion_status(exp):
            print(f"[skip {index}/{len(experiments)}] {exp.name} already complete")
            continue

        print(f"[run {index}/{len(experiments)}] {exp.name}")
        write_metadata(exp)
        if exp.complete_path.exists():
            exp.complete_path.unlink()

        return_code = stream_subprocess(exp.command(), exp.launcher_log_path)
        if return_code == 0:
            exp.complete_path.write_text("ok\n")
        else:
            failures += 1
            print(f"[failed] {exp.name} exited with code {return_code}", file=sys.stderr)
            if args.sleep_between_jobs > 0:
                time.sleep(args.sleep_between_jobs)
            continue

        if args.sleep_between_jobs > 0 and index < len(experiments):
            time.sleep(args.sleep_between_jobs)

    return failures


def parse_metrics(log_path: Path) -> dict[str, float | int | None]:
    best_val_dice = None
    best_test_dice = None
    last_iter = None

    if not log_path.exists():
        return {
            "best_val_dice": None,
            "best_test_dice": None,
            "last_iter": None,
            "completed": False,
        }

    completed = False
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            total_match = TOTAL_ITER_RE.search(line)
            if total_match:
                last_iter = int(total_match.group("iter"))

            val_match = VAL_RE.search(line)
            if val_match:
                current_iter = int(val_match.group("iter"))
                current_dice = float(val_match.group("dice"))
                best_val_dice = current_dice if best_val_dice is None else max(best_val_dice, current_dice)
                last_iter = current_iter
                continue

            test_match = TEST_RE.search(line)
            if test_match:
                current_iter = int(test_match.group("iter"))
                current_dice = float(test_match.group("dice"))
                best_test_dice = current_dice if best_test_dice is None else max(best_test_dice, current_dice)
                last_iter = current_iter
                continue

            if COMPLETE_RE.search(line):
                completed = True

    return {
        "best_val_dice": best_val_dice,
        "best_test_dice": best_test_dice,
        "last_iter": last_iter,
        "completed": completed,
    }


def resolve_key_hparams(exp: ExperimentSpec) -> dict[str, object]:
    values = dict(CONFIG_HPARAM_DEFAULTS[exp.config_name])
    for override in exp.all_overrides():
        if "=" not in override:
            continue
        key, raw_value = override.split("=", 1)
        if key == "model.prototype_value.enable":
            values["prototype_enabled"] = raw_value.lower() == "true"
        elif key == "model.prototype_value.module_mode":
            values["module_mode"] = raw_value
        elif key == "model.prototype_value.mode":
            values["proto_mode"] = raw_value
        elif key == "model.prototype_value.bank.num_proto":
            values["num_proto"] = int(raw_value)
        elif key == "model.prototype_value.bank.temperature":
            values["temperature"] = float(raw_value)
        elif key == "model.prototype_value.temporal.momentum":
            values["momentum"] = float(raw_value)
        elif key == "model.prototype_value.fuse.type":
            values["fuse"] = raw_value
        elif key == "model.prototype_value.feature_source":
            values["feature_source"] = raw_value
    return values


def format_summary_txt(rows: list[dict[str, object]]) -> str:
    lines = []
    lines.append("GDKVM Prototype Ablation Summary")
    lines.append(f"generated_at={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    for row in rows:
        lines.append(f"experiment_name={row['experiment_name']}")
        lines.append(f"dataset={row['dataset']} group={row['group']} variant={row['variant']}")
        lines.append(f"config_name={row['config_name']}")
        lines.append(
            "key_hparams="
            f"prototype_enabled={row['prototype_enabled']}, "
            f"module_mode={row['module_mode']}, "
            f"proto_mode={row['proto_mode']}, "
            f"num_proto={row['num_proto']}, "
            f"temperature={row['temperature']}, "
            f"momentum={row['momentum']}, "
            f"fuse={row['fuse']}, "
            f"feature_source={row['feature_source']}"
        )
        lines.append(
            "metrics="
            f"best_val_dice={row['best_val_dice']}, "
            f"best_test_dice={row['best_test_dice']}, "
            f"last_iter={row['last_iter']}, "
            f"completed={row['completed']}"
        )
        lines.append(f"run_dir={row['run_dir']}")
        lines.append(f"overrides={row['overrides']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_metric(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_summary_md(rows: list[dict[str, object]]) -> str:
    lines = []
    lines.append("# GDKVM Prototype Ablation Report")
    lines.append("")
    lines.append(f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Project root: `{PROJECT_ROOT}`")
    lines.append(f"- Total experiments: `{len(rows)}`")
    lines.append("")

    dataset_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for row in rows:
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1
        group_counts[row["group"]] = group_counts.get(row["group"], 0) + 1

    lines.append("## Coverage")
    lines.append("")
    for dataset, count in sorted(dataset_counts.items()):
        lines.append(f"- Dataset `{dataset}`: {count} experiments")
    for group, count in sorted(group_counts.items()):
        lines.append(f"- Group `{group}`: {count} experiments")
    lines.append("")

    for dataset in sorted({row["dataset"] for row in rows}):
        lines.append(f"## Dataset: {dataset}")
        lines.append("")
        lines.append("| Group | Experiment | Config | Proto | Mode | num_proto | temp | momentum | fuse | feature | Val Dice | Test Dice | Iter | Done |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for row in dataset_rows:
            lines.append(
                "| "
                f"{row['group']} | "
                f"{row['experiment_name']} | "
                f"{row['config_name']} | "
                f"{row['prototype_enabled']} | "
                f"{row['module_mode']}/{row['proto_mode']} | "
                f"{format_metric(row['num_proto'])} | "
                f"{format_metric(row['temperature'])} | "
                f"{format_metric(row['momentum'])} | "
                f"{format_metric(row['fuse'])} | "
                f"{format_metric(row['feature_source'])} | "
                f"{format_metric(row['best_val_dice'])} | "
                f"{format_metric(row['best_test_dice'])} | "
                f"{format_metric(row['last_iter'])} | "
                f"{row['completed']} |"
            )
        lines.append("")

    lines.append("## Experiment Details")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['experiment_name']}")
        lines.append("")
        lines.append(f"- Dataset: `{row['dataset']}`")
        lines.append(f"- Group: `{row['group']}`")
        lines.append(f"- Variant: `{row['variant']}`")
        lines.append(f"- Config: `{row['config_name']}`")
        lines.append(
            "- Key params: "
            f"`prototype_enabled={row['prototype_enabled']}`, "
            f"`module_mode={row['module_mode']}`, "
            f"`proto_mode={row['proto_mode']}`, "
            f"`num_proto={row['num_proto']}`, "
            f"`temperature={row['temperature']}`, "
            f"`momentum={row['momentum']}`, "
            f"`fuse={row['fuse']}`, "
            f"`feature_source={row['feature_source']}`"
        )
        lines.append(
            "- Metrics: "
            f"`best_val_dice={format_metric(row['best_val_dice'])}`, "
            f"`best_test_dice={format_metric(row['best_test_dice'])}`, "
            f"`last_iter={format_metric(row['last_iter'])}`, "
            f"`completed={row['completed']}`"
        )
        lines.append(f"- Run dir: `{row['run_dir']}`")
        lines.append(f"- Overrides: `{row['overrides']}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def collect_results(experiments: Iterable[ExperimentSpec], csv_path: Path, txt_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp in experiments:
        metrics = parse_metrics(exp.launcher_log_path)
        hparams = resolve_key_hparams(exp)
        rows.append(
            {
                "experiment_name": exp.name,
                "dataset": exp.dataset,
                "group": exp.group,
                "variant": exp.variant,
                "config_name": exp.config_name,
                "run_dir": exp.run_dir.as_posix(),
                "overrides": " ".join(exp.all_overrides()),
                "prototype_enabled": hparams["prototype_enabled"],
                "module_mode": hparams["module_mode"],
                "proto_mode": hparams["proto_mode"],
                "num_proto": hparams["num_proto"],
                "temperature": hparams["temperature"],
                "momentum": hparams["momentum"],
                "fuse": hparams["fuse"],
                "feature_source": hparams["feature_source"],
                "best_val_dice": metrics["best_val_dice"],
                "best_test_dice": metrics["best_test_dice"],
                "last_iter": metrics["last_iter"],
                "completed": metrics["completed"] or exp.complete_path.exists(),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_name",
                "dataset",
                "group",
                "variant",
                "config_name",
                "run_dir",
                "overrides",
                "prototype_enabled",
                "module_mode",
                "proto_mode",
                "num_proto",
                "temperature",
                "momentum",
                "fuse",
                "feature_source",
                "best_val_dice",
                "best_test_dice",
                "last_iter",
                "completed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    txt_path.write_text(format_summary_txt(rows), encoding="utf-8")
    md_path.write_text(format_summary_md(rows), encoding="utf-8")


def sync_wandb_runs(log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "wandb",
        "sync",
        "--sync-all",
        "--include-offline",
        "--no-include-synced",
        str(WANDB_ROOT),
    ]
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(cmd)}\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.stdout.close()
        return process.wait()


def print_plan(experiments: list[ExperimentSpec]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for exp in experiments:
        counts[(exp.group, exp.dataset)] = counts.get((exp.group, exp.dataset), 0) + 1

    print("Selected experiments:")
    for group in ["core", "ablation", "tune"]:
        group_total = sum(counts.get((group, dataset), 0) for dataset in sorted(DATASET_SPECS))
        if group_total == 0:
            continue
        per_dataset = ", ".join(
            f"{dataset}={counts.get((group, dataset), 0)}" for dataset in sorted(DATASET_SPECS)
        )
        print(f"  {group}: {group_total} ({per_dataset})")
    print(f"  total: {len(experiments)}")


def main() -> int:
    args = parse_args()
    experiments = select_experiments(args)

    if not experiments:
        print("No experiments selected.", file=sys.stderr)
        return 1

    print_plan(experiments)
    write_exports(experiments, args.export_txt, args.export_sh)

    for exp in experiments:
        print(exp.shell_command())

    should_collect = args.collect_results or args.run
    should_sync_wandb = (args.run and not args.no_sync_wandb) or args.sync_wandb
    if args.run:
        failures = run_experiments(experiments, args)
        collect_results(experiments, args.summary_csv, args.summary_txt, args.summary_md)
        sync_failures = 0
        if should_sync_wandb:
            sync_failures = sync_wandb_runs(args.wandb_sync_log)
        print(f"Summary CSV: {args.summary_csv}")
        print(f"Summary TXT: {args.summary_txt}")
        print(f"Summary MD: {args.summary_md}")
        if should_sync_wandb:
            print(f"W&B sync log: {args.wandb_sync_log}")
        return 1 if (failures or sync_failures) else 0

    if args.dry_run or should_collect or should_sync_wandb:
        collect_results(experiments, args.summary_csv, args.summary_txt, args.summary_md)
        if should_collect:
            print(f"Summary CSV: {args.summary_csv}")
            print(f"Summary TXT: {args.summary_txt}")
            print(f"Summary MD: {args.summary_md}")
        if should_sync_wandb:
            sync_wandb_runs(args.wandb_sync_log)
            print(f"W&B sync log: {args.wandb_sync_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
