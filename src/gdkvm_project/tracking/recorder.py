from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf


class RunRecorder:
    """Local, MLflow-independent artifact recorder.

    The recorder writes a stable set of files for every run so experiments stay
    inspectable even when remote tracking is disabled or temporarily unavailable.
    """

    REQUIRED_FILES = (
        "config_resolved.yaml",
        "runtime.json",
        "git.json",
        "metrics.jsonl",
        "summary.json",
        "data_flow_summary.json",
    )

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.metrics_path.touch(exist_ok=True)

    def log_config(self, cfg: DictConfig | Mapping[str, Any]) -> Path:
        path = self.run_dir / "config_resolved.yaml"
        if isinstance(cfg, (DictConfig, ListConfig)):
            content = OmegaConf.to_yaml(cfg, resolve=True)
        else:
            content = json.dumps(cfg, indent=2, sort_keys=True, default=str)
        path.write_text(content, encoding="utf-8")
        return path

    def log_runtime(self, extra: Mapping[str, Any] | None = None) -> Path:
        payload = {
            "time_unix": time.time(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
        }
        if extra:
            payload.update(dict(extra))
        return self._write_json("runtime.json", payload)

    def log_git(self) -> Path:
        root = Path(__file__).resolve().parents[3]

        def run_git(args: list[str], default: Any) -> Any:
            try:
                return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
            except Exception:
                return default

        payload = {
            "commit": run_git(["rev-parse", "HEAD"], "unknown"),
            "short": run_git(["rev-parse", "--short", "HEAD"], "nogit"),
            "dirty": bool(run_git(["status", "--short"], "")),
        }
        return self._write_json("git.json", payload)

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None, split: str | None = None) -> None:
        row = {"time_unix": time.time(), "step": step, "split": split, "metrics": dict(metrics)}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def log_summary(self, summary: Mapping[str, Any]) -> Path:
        return self._write_json("summary.json", dict(summary))

    def log_data_flow_summary(self, summary: Mapping[str, Any]) -> Path:
        return self._write_json("data_flow_summary.json", dict(summary))

    def ensure_required_files(self) -> None:
        for name in self.REQUIRED_FILES:
            path = self.run_dir / name
            if not path.exists():
                if name.endswith(".jsonl"):
                    path.touch()
                elif name.endswith(".yaml"):
                    path.write_text("{}\n", encoding="utf-8")
                else:
                    self._write_json(name, {})

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return path
