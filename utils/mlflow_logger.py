from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf


class MLflowLogger:
    """Single MLflow access layer for experiment tracking."""

    def __init__(
        self,
        cfg: Mapping[str, Any] | DictConfig | None,
        *,
        run_dir: str | Path,
        enabled: bool = True,
        main_process: bool = True,
    ) -> None:
        self.cfg = cfg or {}
        self.run_dir = Path(run_dir)
        self.enabled = bool(enabled) and bool(main_process)
        self.main_process = bool(main_process)
        self.run = None
        self.run_id = None

    def start_run(self) -> None:
        if not self.enabled:
            return
        import mlflow

        tracking_uri = str(self._cfg_get("tracking_uri", "http://172.16.240.77:5000"))
        experiment_name = str(self._cfg_get("experiment_name", "anchor_ode"))
        run_name = self._cfg_get("run_name", None)
        resume_run_id = self._cfg_get("resume_run_id", None)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        kwargs: dict[str, Any] = {}
        if resume_run_id:
            kwargs["run_id"] = str(resume_run_id)
        elif run_name:
            kwargs["run_name"] = str(run_name)
        self.run = mlflow.start_run(**kwargs)
        self.run_id = getattr(getattr(self.run, "info", None), "run_id", None)

    def end_run(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        import mlflow

        mlflow.end_run(status=status)

    def mark_failed(self) -> None:
        self.end_run(status="FAILED")

    def log_config(self, cfg: DictConfig | Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.run_dir / "resolved_config.yaml"
        if isinstance(cfg, (DictConfig, ListConfig)):
            content = OmegaConf.to_yaml(cfg, resolve=True)
            container = OmegaConf.to_container(cfg, resolve=True)
        else:
            content = json.dumps(cfg, indent=2, sort_keys=True, default=str)
            container = cfg
        config_path.write_text(content, encoding="utf-8")
        self.log_artifact(config_path, artifact_path="configs")
        self.log_params(self._flatten(container))

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        import mlflow

        clean = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple, set)):
                value = json.dumps(value, sort_keys=True, default=str)
            clean[str(key)] = str(value)[:500]
        for start in range(0, len(clean), 100):
            chunk = dict(list(clean.items())[start : start + 100])
            if chunk:
                try:
                    mlflow.log_params(chunk)
                except Exception:
                    for key, value in chunk.items():
                        try:
                            mlflow.log_param(key, value)
                        except Exception:
                            continue

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None, prefix: str | None = None) -> None:
        if not self.enabled:
            return
        import mlflow

        clean = {}
        for key, value in metrics.items():
            scalar = self._to_float(value)
            if scalar is None or not math.isfinite(scalar):
                continue
            name = f"{prefix}/{key}" if prefix else str(key)
            clean[name] = scalar
        if clean:
            mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        import mlflow

        path = Path(path)
        if path.exists() and path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        import mlflow

        path = Path(path)
        if path.exists() and path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)

    def log_checkpoint(self, path: str | Path, *, name: str | None = None) -> None:
        path = Path(path)
        artifact_path = "checkpoints" if name is None else f"checkpoints/{name}"
        self.log_artifact(path, artifact_path=artifact_path)

    def log_evaluation_result(self, result: Any, *, step: int | None = None) -> None:
        if not self.enabled or result is None:
            return
        mode = str(getattr(result, "mode", "eval"))
        summary_metrics = dict(getattr(result, "summary_metrics", {}) or {})
        self.log_metrics(summary_metrics, step=step, prefix=mode)

        with tempfile.TemporaryDirectory(prefix="mlflow_eval_") as tmp:
            tmp_path = Path(tmp)
            summary_path = tmp_path / f"{mode}_summary.json"
            summary_path.write_text(json.dumps(summary_metrics, indent=2, sort_keys=True), encoding="utf-8")
            self.log_artifact(summary_path, artifact_path="eval")

            threshold_sweep = dict(getattr(result, "threshold_sweep", {}) or {})
            if threshold_sweep:
                sweep_path = tmp_path / f"{mode}_threshold_sweep.json"
                sweep_path.write_text(json.dumps(threshold_sweep, indent=2, sort_keys=True), encoding="utf-8")
                self.log_artifact(sweep_path, artifact_path="eval")

            for attr, filename in (
                ("per_video_metrics", f"{mode}_per_video.csv"),
                ("per_frame_metrics", f"{mode}_per_frame.csv"),
            ):
                rows = list(getattr(result, attr, []) or [])
                if rows:
                    csv_path = tmp_path / filename
                    self._write_csv(csv_path, rows)
                    self.log_artifact(csv_path, artifact_path="eval")

        for artifact in getattr(result, "visual_artifacts", []) or []:
            self.log_artifact(artifact, artifact_path="visuals")

    def log_env_info(self) -> None:
        if not self.enabled:
            return
        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        try:
            import torch

            env["torch"] = torch.__version__
            env["cuda_available"] = torch.cuda.is_available()
            env["cuda_device_count"] = torch.cuda.device_count()
        except Exception as exc:
            env["torch_error"] = str(exc)
        self._log_json_artifact(env, "env/runtime.json", artifact_path="env")

    def log_git_info(self) -> None:
        if not self.enabled:
            return
        root = Path(__file__).resolve().parents[1]

        def git(*args: str) -> str:
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

        info: dict[str, Any] = {}
        try:
            info["commit"] = git("rev-parse", "HEAD")
            info["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
            info["status_short"] = git("status", "--short")
        except Exception as exc:
            info["error"] = str(exc)
        self._log_json_artifact(info, "source/git.json", artifact_path="source")

    def _cfg_get(self, key: str, default: Any) -> Any:
        if hasattr(self.cfg, "get"):
            return self.cfg.get(key, default)
        return default

    def _log_json_artifact(self, payload: Mapping[str, Any], filename: str, *, artifact_path: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        self.log_artifact(path, artifact_path=artifact_path)

    @staticmethod
    def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, Mapping):
            out = {}
            for key, item in value.items():
                next_key = f"{prefix}.{key}" if prefix else str(key)
                out.update(MLflowLogger._flatten(item, next_key))
            return out
        if isinstance(value, (list, tuple)):
            return {prefix: json.dumps(value, default=str)}
        return {prefix: value}

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "mean"):
                value = value.float().mean()
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
