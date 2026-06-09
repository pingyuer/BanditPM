from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf


class MLflowLogger:
    """Single MLflow access layer for experiment tracking."""

    PARAM_ROOTS = {
        "model",
        "data",
        "evaluation",
        "main_training",
        "loss",
        "losses",
        "dataset_name",
        "data_path",
        "exp_id",
        "model_name",
        "seed",
    }

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
        self.required = bool(self._cfg_get("required", True))
        self.artifacts_required = bool(self._cfg_get("artifacts_required", True))
        self.artifacts_enabled = bool(self._cfg_get("artifacts_enabled", True))
        self.timeout_seconds = int(self._cfg_get("timeout_seconds", 30) or 30)
        self.run = None
        self.run_id = None

    def start_run(self, *, tags: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        def _start():
            import mlflow

            tracking_uri = str(self._cfg_get("tracking_uri", None) or "http://172.16.240.77:5000")
            experiment_name = str(self._cfg_get("experiment_name", None) or "experiment")
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
            if tags:
                self.log_tags(tags)

        self._call(_start, "start_run", required=self.required)

    def preflight(self) -> None:
        if not self.enabled or not bool(self._cfg_get("preflight", True)):
            return

        def _probe():
            import mlflow

            tracking_uri = str(self._cfg_get("tracking_uri", None) or "http://172.16.240.77:5000")
            experiment_name = str(self._cfg_get("experiment_name", None) or "experiment")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            run_name = f"preflight_{int(time.time())}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("run_type", "preflight")
                mlflow.set_tag("stage", str(self._cfg_get("stage", "full")))
                mlflow.log_metric("preflight/alive", 1.0, step=0)
                if self.artifacts_enabled or self.artifacts_required:
                    with tempfile.TemporaryDirectory(prefix="mlflow_preflight_") as tmp:
                        path = Path(tmp) / "preflight.txt"
                        path.write_text("ok\n", encoding="utf-8")
                        mlflow.log_artifact(str(path), artifact_path="env")

        self._call(_probe, "preflight", required=True)

    def start_eval_run(
        self,
        *,
        source_run_id: str,
        source_checkpoint: str,
        eval_mode: str,
        dataset: str | None = None,
        protocol: str | None = None,
    ) -> None:
        tags = {
            "run_type": "eval",
            "source_run_id": source_run_id,
            "source_checkpoint": source_checkpoint,
            "eval_mode": eval_mode,
        }
        if dataset:
            tags["dataset"] = dataset
        if protocol:
            tags["protocol"] = protocol
        self.start_run(tags=tags)

    def end_run(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        def _end():
            import mlflow

            mlflow.end_run(status=status)

        self._call(_end, "end_run", required=False)

    def mark_failed(self) -> None:
        self.end_run(status="FAILED")

    def log_config(self, cfg: DictConfig | Mapping[str, Any], *, overrides: list[str] | None = None) -> None:
        if not self.enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(cfg, (DictConfig, ListConfig)):
            raw_content = OmegaConf.to_yaml(cfg, resolve=False)
            resolved_content = OmegaConf.to_yaml(cfg, resolve=True)
        else:
            raw_content = json.dumps(cfg, indent=2, sort_keys=True, default=str)
            resolved_content = raw_content
        config_path = self.run_dir / "config.yaml"
        resolved_path = self.run_dir / "config_resolved.yaml"
        overrides_path = self.run_dir / "overrides.txt"
        config_path.write_text(raw_content, encoding="utf-8")
        resolved_path.write_text(resolved_content, encoding="utf-8")
        overrides_path.write_text("\n".join(overrides or []) + ("\n" if overrides else ""), encoding="utf-8")
        self.log_artifact(config_path, artifact_path="configs")
        self.log_artifact(resolved_path, artifact_path="configs")
        self.log_artifact(overrides_path, artifact_path="configs")

    def log_run_metadata(
        self,
        *,
        tags: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        if tags:
            self.log_tags(tags)
        if params:
            self.log_params(params)

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        def _params():
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
                    mlflow.log_params(chunk)

        self._call(_params, "log_params", required=self.required)

    def log_tags(self, tags: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        clean = {str(key): str(value) for key, value in tags.items() if value is not None}
        if not clean:
            return

        def _tags():
            import mlflow

            mlflow.set_tags(clean)

        self._call(_tags, "log_tags", required=self.required)

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None, prefix: str | None = None) -> None:
        if not self.enabled:
            return
        clean = {}
        for key, value in metrics.items():
            scalar = self._to_float(value)
            if scalar is None or not math.isfinite(scalar):
                continue
            name = f"{prefix}/{key}" if prefix else str(key)
            clean[name] = scalar
        if clean:
            def _metrics():
                import mlflow

                mlflow.log_metrics(clean, step=step)

            self._call(_metrics, "log_metrics", required=self.required)

    def log_train_step(self, metrics: Mapping[str, Any], *, step: int) -> None:
        mapping = {
            "total_loss": "loss/total",
            "loss": "loss/total",
            "seg_loss": "loss/seg",
            "dice_loss": "loss/dice",
            "bce_loss": "loss/bce",
            "base_loss": "loss/base",
            "base_seg_loss": "loss/base",
            "guided_loss": "loss/guided",
            "guided_seg_loss": "loss/guided",
            "prior_loss": "loss/prior",
            "prior_seg_loss": "loss/prior",
            "conf_loss": "loss/conf",
            "confidence_loss": "loss/conf",
            "lr": "lr",
            "residual_head_lr": "residual_head_lr",
            "anchor_temperature": "anchor_temperature",
            "residual_scale": "residual_scale",
        }
        out = {}
        for key, value in metrics.items():
            key_str = str(key)
            if key_str in mapping:
                out[mapping[key_str]] = value
            elif key_str.startswith("loss/"):
                out[key_str] = value
            elif key_str.startswith("aux_functional_anchor_"):
                out[f"loss/weighted/functional_anchor/{key_str.removeprefix('aux_functional_anchor_')}"] = value
            elif key_str.startswith("raw_functional_anchor_"):
                out[f"loss/raw/functional_anchor/{key_str.removeprefix('raw_functional_anchor_')}"] = value
            elif key_str.startswith("lambda_functional_anchor_"):
                out[f"lambda/functional_anchor/{key_str.removeprefix('lambda_functional_anchor_')}"] = value
            elif key_str.startswith("aux_faf_"):
                out[f"loss/weighted/faf/{key_str.removeprefix('aux_faf_')}"] = value
            elif key_str.startswith("raw_faf_"):
                out[f"loss/raw/faf/{key_str.removeprefix('raw_faf_')}"] = value
            elif key_str.startswith("lambda_faf_"):
                out[f"lambda/faf/{key_str.removeprefix('lambda_faf_')}"] = value
            elif key_str.startswith("aux_gar_"):
                out[f"loss/weighted/gar/{key_str.removeprefix('aux_gar_')}"] = value
            elif key_str.startswith("raw_gar_"):
                out[f"loss/raw/gar/{key_str.removeprefix('raw_gar_')}"] = value
            elif key_str.startswith("lambda_gar_"):
                out[f"lambda/gar/{key_str.removeprefix('lambda_gar_')}"] = value
            elif key_str.startswith("aux_cardia_"):
                out[f"loss/weighted/cardia/{key_str.removeprefix('aux_cardia_')}"] = value
            elif key_str.startswith("raw_cardia_"):
                out[f"loss/raw/cardia/{key_str.removeprefix('raw_cardia_')}"] = value
            elif key_str.startswith("lambda_cardia_"):
                out[f"lambda/cardia/{key_str.removeprefix('lambda_cardia_')}"] = value
        self.log_metrics(out, step=step, prefix="train")

    def log_eval_summary(self, metrics: Mapping[str, Any], *, mode: str, step: int | None = None) -> None:
        out = self._standard_eval_metrics(metrics)
        self.log_metrics(out, step=step, prefix=mode)
        functional = self._functional_anchor_metrics(metrics)
        if functional:
            self.log_metrics(functional, step=step, prefix=f"{mode}/functional_anchor")
        faf = self._faf_metrics(metrics)
        if faf:
            self.log_metrics(faf, step=step, prefix=f"{mode}/faf")
        anchor = self._anchor_ode_metrics(metrics, require_explicit=True)
        if anchor:
            self.log_metrics(anchor, step=step, prefix=f"{mode}/anchor_ode")
        gar = self._gar_metrics(metrics)
        if gar:
            self.log_metrics(gar, step=step, prefix=f"{mode}/gar")
        cardia = self._cardia_metrics(metrics)
        if cardia:
            self.log_metrics(cardia, step=step, prefix=f"{mode}/cardia")

    def log_best(self, metrics: Mapping[str, Any], *, epoch: int, iteration: int) -> None:
        best = {
            "val_dice": metrics.get("dice", metrics.get("dice_frame_mean")),
            "val_iou": metrics.get("iou", metrics.get("iou_frame_mean")),
            "val_hd95": metrics.get("hd95", metrics.get("hd95_original", metrics.get("hd95_resized"))),
            "epoch": epoch,
            "iter": iteration,
        }
        self.log_metrics(best, step=iteration, prefix="best")

    def log_anchor_ode_diagnostics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        self.log_metrics(self._anchor_ode_metrics(metrics, require_explicit=False), step=step, prefix="anchor_ode")

    @classmethod
    def _anchor_ode_metrics(cls, metrics: Mapping[str, Any], *, require_explicit: bool = True) -> dict[str, Any]:
        if require_explicit and not any(str(key).startswith("anchor_ode/") for key in metrics):
            return {}
        out = {}
        aliases = {
            "base_dice": ("base_dice", "base_only_dice_frame_mean", "anchor_ode/base_dice"),
            "guided_dice": ("guided_dice", "guided_only_dice_frame_mean", "anchor_ode/guided_dice"),
            "final_dice": ("final_dice", "dice_frame_mean", "dice", "anchor_ode/final_dice"),
            "prior_dice": ("prior_dice", "prior_only_dice_frame_mean", "anchor_ode/prior_dice"),
            "gate_mean": ("gate_mean", "anchor_ode/gate_mean"),
            "gate_std": ("gate_std", "anchor_ode/gate_std"),
            "confidence_prior_mean": ("confidence_prior_mean", "confidence_prior", "anchor_ode/confidence_prior"),
            "confidence_update_mean": ("confidence_update_mean", "confidence_update", "anchor_ode/confidence_update"),
            "residual_abs_mean": ("residual_abs_mean", "anchor_ode/residual_abs_mean"),
            "final_base_residual_abs_mean": ("final_base_residual_abs_mean", "anchor_ode/final_base_residual_abs_mean"),
            "guided_base_residual_abs_mean": ("guided_base_residual_abs_mean", "anchor_ode/guided_base_residual_abs_mean"),
            "affine_abs_mean": ("affine_abs_mean", "anchor_ode/affine_abs_mean"),
            "translate_abs_mean": ("translate_abs_mean", "anchor_ode/translate_abs_mean"),
            "scale_abs_mean": ("scale_abs_mean", "anchor_ode/scale_abs_mean"),
            "rotate_abs_mean": ("rotate_abs_mean", "anchor_ode/rotate_abs_mean"),
            "shear_abs_mean": ("shear_abs_mean", "anchor_ode/shear_abs_mean"),
            "slot_entropy": ("slot_entropy", "anchor_ode/slot_entropy"),
            "slot_max_prob": ("slot_max_prob", "anchor_ode/slot_max_prob"),
        }
        for dst, keys in aliases.items():
            for key in keys:
                if key in metrics:
                    out[dst] = metrics[key]
                    break
        if "final_dice" in out and "base_dice" in out:
            final = cls._to_float(out["final_dice"])
            base = cls._to_float(out["base_dice"])
            if final is not None and base is not None:
                out["final_minus_base_dice"] = final - base
        if "guided_dice" in out and "base_dice" in out:
            guided = cls._to_float(out["guided_dice"])
            base = cls._to_float(out["base_dice"])
            if guided is not None and base is not None:
                out["guided_minus_base_dice"] = guided - base
        return out

    def log_functional_anchor_diagnostics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        self.log_metrics(self._functional_anchor_metrics(metrics), step=step, prefix="functional_anchor")

    def log_faf_diagnostics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        self.log_metrics(self._faf_metrics(metrics), step=step, prefix="faf")

    def log_gar_diagnostics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        self.log_metrics(self._gar_metrics(metrics), step=step, prefix="gar")

    def log_cardia_diagnostics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        self.log_metrics(self._cardia_metrics(metrics), step=step, prefix="cardia")

    @classmethod
    def _cardia_metrics(cls, metrics: Mapping[str, Any]) -> dict[str, Any]:
        aliases = {
            "base_dice": ("base_dice", "cardia/base_dice"),
            "proposal_oracle": ("proposal_oracle_dice", "cardia/proposal_oracle"),
            "proposal_top1": ("proposal_top1_dice", "cardia/proposal_top1"),
            "proposal_oracle_minus_top1": ("proposal_oracle_minus_top1", "cardia/proposal_oracle_minus_top1"),
            "proposal_top1_minus_base": ("proposal_top1_minus_base", "cardia/proposal_top1_minus_base"),
            "selector_oracle_alignment": ("selector_oracle_alignment", "cardia/selector_oracle_alignment"),
            "final_dice": ("final_dice", "dice_frame_mean", "dice", "cardia/final_dice"),
            "boundary_dice": ("boundary_dice", "boundary_dice_frame_mean", "cardia/boundary_dice"),
            "final_minus_base": ("final_minus_base_dice", "final_minus_base", "cardia/final_minus_base"),
            "stage3/flow_smooth": ("stage3_flow_smooth", "cardia/stage3/flow_smooth"),
            "stage3/write_mean": ("stage3_write_mean", "cardia/stage3/write_mean"),
            "stage3/decay_mean": ("stage3_decay_mean", "cardia/stage3/decay_mean"),
            "stage3/gamma": ("stage3_gamma", "cardia/stage3/gamma"),
            "stage3/runtime_update_mean": ("stage3_runtime_update_mean", "cardia/stage3/runtime_update_mean"),
            "stage3/runtime_state_norm": ("stage3_runtime_state_norm", "cardia/stage3/runtime_state_norm"),
            "stage2/flow_smooth": ("stage2_flow_smooth", "cardia/stage2/flow_smooth"),
            "stage2/write_mean": ("stage2_write_mean", "cardia/stage2/write_mean"),
            "stage2/decay_mean": ("stage2_decay_mean", "cardia/stage2/decay_mean"),
            "stage2/gamma": ("stage2_gamma", "cardia/stage2/gamma"),
            "stage2/global_selector_entropy": ("stage2_global_selector_entropy", "cardia/stage2/global_selector_entropy"),
            "stage2/head_usage_entropy": ("stage2_head_usage_entropy", "cardia/stage2/head_usage_entropy"),
            "stage2/selector_logit_scale": ("stage2_selector_logit_scale", "cardia/stage2/selector_logit_scale"),
            "stage2/runtime_update_mean": ("stage2_runtime_update_mean", "cardia/stage2/runtime_update_mean"),
            "solver/offset_px_mean": ("solver_offset_px_mean", "cardia/solver/offset_px_mean"),
            "solver/offset_px_p95": ("solver_offset_px_p95", "cardia/solver/offset_px_p95"),
            "boundary/edge_gate": ("boundary_edge_gate_mean", "cardia/boundary/edge_gate"),
            "boundary/edge_gate_p05": ("boundary_edge_gate_p05", "cardia/boundary/edge_gate_p05"),
            "boundary/edge_gate_p95": ("boundary_edge_gate_p95", "cardia/boundary/edge_gate_p95"),
            "boundary/channel_gate_mean": ("boundary_channel_gate_mean", "cardia/boundary/channel_gate_mean"),
            "boundary/delta_abs_mean": ("boundary_delta_abs_mean", "cardia/boundary/delta_abs_mean"),
            "boundary/inside_response": ("boundary_inside_response", "cardia/boundary/inside_response"),
            "boundary/outside_response": ("boundary_outside_response", "cardia/boundary/outside_response"),
        }
        out = {}
        for dst, keys in aliases.items():
            for key in keys:
                if key in metrics:
                    out[dst] = metrics[key]
                    break
        for stage in ("stage2", "stage3"):
            for idx in range(16):
                for key in (f"{stage}_head_usage_{idx}", f"cardia/{stage}/head_usage_{idx}"):
                    if key in metrics:
                        out[f"{stage}/head_usage_{idx}"] = metrics[key]
                        break
        if "final_minus_base" not in out and "final_dice" in out and "base_dice" in out:
            final = cls._to_float(out["final_dice"])
            base = cls._to_float(out["base_dice"])
            if final is not None and base is not None:
                out["final_minus_base"] = final - base
        if not any(str(key).startswith("cardia/") for key in metrics) and not any(
            key in metrics for key in ("stage2_flow_smooth", "proposal_oracle_dice")
        ):
            return {}
        return out

    @classmethod
    def _gar_metrics(cls, metrics: Mapping[str, Any]) -> dict[str, Any]:
        aliases = {
            "base_dice": ("base_dice", "gar/base_dice"),
            "proposal_oracle_dice": ("proposal_oracle_dice", "gar/proposal_oracle_dice"),
            "proposal_top1_dice": ("proposal_top1_dice", "gar/proposal_top1_dice"),
            "proposal_oracle_minus_top1": ("proposal_oracle_minus_top1", "gar/proposal_oracle_minus_top1"),
            "proposal_top1_minus_base": ("proposal_top1_minus_base", "gar/proposal_top1_minus_base"),
            "selector_oracle_alignment": ("selector_oracle_alignment", "gar/selector_oracle_alignment"),
            "final_dice": ("final_dice", "dice_frame_mean", "dice", "gar/final_dice"),
            "boundary_dice": ("boundary_dice", "boundary_dice_frame_mean", "gar/boundary_dice"),
            "final_minus_base_dice": ("final_minus_base_dice", "final_minus_base", "gar/final_minus_base_dice"),
            "final_minus_base_by_ED": ("final_minus_base_by_ED", "gar/final_minus_base_by_ED"),
            "final_minus_base_by_ES": ("final_minus_base_by_ES", "gar/final_minus_base_by_ES"),
            "stage3_offset_px_mean": ("stage3_offset_px_mean", "gar/stage3_offset_px_mean"),
            "stage3_offset_px_p95": ("stage3_offset_px_p95", "gar/stage3_offset_px_p95"),
            "stage3_flow_smooth": ("stage3_flow_smooth", "gar/stage3_flow_smooth"),
            "stage3_write_mean": ("stage3_write_mean", "gar/stage3_write_mean"),
            "stage3_write_p05": ("stage3_write_p05", "gar/stage3_write_p05"),
            "stage3_write_p95": ("stage3_write_p95", "gar/stage3_write_p95"),
            "stage3_decay_mean": ("stage3_decay_mean", "gar/stage3_decay_mean"),
            "stage3_gamma": ("stage3_gamma", "gar/stage3_gamma"),
            "stage3_selector_logit_scale": ("stage3_selector_logit_scale", "gar/stage3_selector_logit_scale"),
            "stage3_head_entropy": ("stage3_head_entropy", "gar/stage3_head_entropy"),
            "stage3_global_selector_entropy": ("stage3_global_selector_entropy", "gar/stage3_global_selector_entropy"),
            "stage3_head_usage_entropy": ("stage3_head_usage_entropy", "gar/stage3_head_usage_entropy"),
            "stage3_head_usage_max": ("stage3_head_usage_max", "gar/stage3_head_usage_max"),
            "stage3_head_usage_min": ("stage3_head_usage_min", "gar/stage3_head_usage_min"),
            "stage3_head_max_weight": ("stage3_head_max_weight", "gar/stage3_head_max_weight"),
            "stage2_offset_px_mean": ("stage2_offset_px_mean", "gar/stage2_offset_px_mean"),
            "stage2_offset_px_p95": ("stage2_offset_px_p95", "gar/stage2_offset_px_p95"),
            "stage2_flow_smooth": ("stage2_flow_smooth", "gar/stage2_flow_smooth"),
            "stage2_write_mean": ("stage2_write_mean", "gar/stage2_write_mean"),
            "stage2_write_p05": ("stage2_write_p05", "gar/stage2_write_p05"),
            "stage2_write_p95": ("stage2_write_p95", "gar/stage2_write_p95"),
            "stage2_decay_mean": ("stage2_decay_mean", "gar/stage2_decay_mean"),
            "stage2_gamma": ("stage2_gamma", "gar/stage2_gamma"),
            "stage2_selector_logit_scale": ("stage2_selector_logit_scale", "gar/stage2_selector_logit_scale"),
            "stage2_head_entropy": ("stage2_head_entropy", "gar/stage2_head_entropy"),
            "stage2_global_selector_entropy": ("stage2_global_selector_entropy", "gar/stage2_global_selector_entropy"),
            "stage2_head_usage_entropy": ("stage2_head_usage_entropy", "gar/stage2_head_usage_entropy"),
            "stage2_head_usage_max": ("stage2_head_usage_max", "gar/stage2_head_usage_max"),
            "stage2_head_usage_min": ("stage2_head_usage_min", "gar/stage2_head_usage_min"),
            "stage2_head_max_weight": ("stage2_head_max_weight", "gar/stage2_head_max_weight"),
            "boundary_gamma": ("boundary_gamma", "gar/boundary_gamma"),
            "boundary_gate_mean": ("boundary_gate_mean", "gar/boundary_gate_mean"),
            "boundary_edge_gate_mean": ("boundary_edge_gate_mean", "gar/boundary_edge_gate_mean"),
            "boundary_edge_gate_p05": ("boundary_edge_gate_p05", "gar/boundary_edge_gate_p05"),
            "boundary_edge_gate_p95": ("boundary_edge_gate_p95", "gar/boundary_edge_gate_p95"),
            "boundary_channel_gate_mean": ("boundary_channel_gate_mean", "gar/boundary_channel_gate_mean"),
            "boundary_inside_response": ("boundary_inside_response", "gar/boundary_inside_response"),
            "boundary_outside_response": ("boundary_outside_response", "gar/boundary_outside_response"),
            "boundary_delta_abs_mean": ("boundary_delta_abs_mean", "gar/boundary_delta_abs_mean"),
            "boundary_raw_delta_abs_mean": ("boundary_raw_delta_abs_mean", "gar/boundary_raw_delta_abs_mean"),
        }
        out = {}
        for dst, keys in aliases.items():
            for key in keys:
                if key in metrics:
                    out[dst] = metrics[key]
                    break
        for stage in ("stage2", "stage3"):
            for idx in range(16):
                for key in (f"{stage}_head_usage_{idx}", f"gar/{stage}_head_usage_{idx}"):
                    if key in metrics:
                        out[f"{stage}_head_usage_{idx}"] = metrics[key]
                        break
        if "final_minus_base_dice" not in out and "final_dice" in out and "base_dice" in out:
            final = cls._to_float(out["final_dice"])
            base = cls._to_float(out["base_dice"])
            if final is not None and base is not None:
                out["final_minus_base_dice"] = final - base
        if not any(str(key).startswith("gar/") for key in metrics) and not any(
            key in metrics for key in ("stage2_offset_px_mean", "proposal_oracle_dice")
        ):
            return {}
        return out

    @classmethod
    def _faf_metrics(cls, metrics: Mapping[str, Any]) -> dict[str, Any]:
        aliases = {
            "base_dice": ("base_dice", "faf/base_dice"),
            "unext_anchor_dice": ("unext_anchor_dice", "faf/unext_anchor_dice"),
            "affine_identity_dice": ("affine_identity_dice", "faf/affine_identity_dice"),
            "affine_mixture_dice": ("affine_mixture_dice", "faf/affine_mixture_dice"),
            "affine_top1_dice": ("affine_top1_dice", "faf/affine_top1_dice"),
            "affine_oracle_dice": ("affine_oracle_dice", "faf/affine_oracle_dice"),
            "affine_mean_dice": ("affine_mean_dice", "faf/affine_mean_dice"),
            "final_dice": ("final_dice", "dice_frame_mean", "dice", "faf/final_dice"),
            "final_minus_base_dice": ("final_minus_base_dice", "final_minus_base", "faf/final_minus_base_dice"),
            "oracle_gap_to_base": ("oracle_gap_to_base", "faf/oracle_gap_to_base"),
            "final_gap_to_base": ("final_gap_to_base", "faf/final_gap_to_base"),
            "final_below_base_alert": ("final_below_base_alert", "faf/final_below_base_alert"),
            "final_minus_base_by_ED": ("final_minus_base_by_ED", "faf/final_minus_base_by_ED"),
            "final_minus_base_by_ES": ("final_minus_base_by_ES", "faf/final_minus_base_by_ES"),
            "hard_frame_final_minus_base": ("hard_frame_final_minus_base", "faf/hard_frame_final_minus_base"),
            "area_curve_corr": ("area_curve_corr", "faf/area_curve_corr"),
            "temporal_jitter_delta": ("temporal_jitter_delta", "faf/temporal_jitter_delta"),
            "effective_slot_number": ("effective_slot_number", "faf/effective_slot_number"),
            "slot_entropy": ("slot_entropy", "faf/slot_entropy"),
            "top1_slot_weight": ("top1_slot_weight", "faf/top1_slot_weight"),
            "top3_slot_weight_sum": ("top3_slot_weight_sum", "faf/top3_slot_weight_sum"),
            "coverage_score": ("coverage_score", "faf/coverage_score"),
            "coverage_gap": ("coverage_gap", "faf/coverage_gap"),
            "slot_area_diversity": ("slot_area_diversity", "faf/slot_area_diversity"),
            "write_strength_mean": ("write_strength_mean", "faf/write_strength_mean"),
            "memory_update_norm": ("memory_update_norm", "faf/memory_update_norm"),
            "affine_delta_norm": ("affine_delta_norm", "faf/affine_delta_norm"),
            "affine_state_norm": ("affine_state_norm", "faf/affine_state_norm"),
            "velocity_norm": ("velocity_norm", "faf/velocity_norm"),
            "confidence_mean": ("confidence_mean", "faf/confidence_mean"),
            "confidence_easy_mean": ("confidence_easy_mean", "faf/confidence_easy_mean"),
            "confidence_hard_mean": ("confidence_hard_mean", "faf/confidence_hard_mean"),
            "identity_slot_usage": ("identity_slot_usage", "faf/identity_slot_usage"),
            "residual_l1": ("residual_l1", "faf/residual_l1"),
            "residual_l2": ("residual_l2", "faf/residual_l2"),
            "safety_residual_l1": ("safety_residual_l1", "faf/safety_residual_l1"),
            "residual_clip_hit_ratio": ("residual_clip_hit_ratio", "faf/residual_clip_hit_ratio"),
            "residual_scale": ("residual_scale", "faf/residual_scale"),
            "retrieval_temperature": ("retrieval_temperature", "faf/retrieval_temperature"),
            "ode_dt": ("ode_dt", "faf/ode_dt"),
            "feature_modulation_l1": ("feature_modulation_l1", "faf/feature_modulation_l1"),
            "feature_modulation_l1_low": ("feature_modulation_l1_low", "faf/feature_modulation_l1_low"),
            "feature_modulation_l1_mid": ("feature_modulation_l1_mid", "faf/feature_modulation_l1_mid"),
            "feature_modulation_l1_high": ("feature_modulation_l1_high", "faf/feature_modulation_l1_high"),
            "feature_modulation_l1_dec": ("feature_modulation_l1_dec", "faf/feature_modulation_l1_dec"),
        }
        out = {}
        for dst, keys in aliases.items():
            for key in keys:
                if key in metrics:
                    out[dst] = metrics[key]
                    break
        if "final_minus_base_dice" not in out and "final_dice" in out and "base_dice" in out:
            final = cls._to_float(out["final_dice"])
            base = cls._to_float(out["base_dice"])
            if final is not None and base is not None:
                out["final_minus_base_dice"] = final - base
        if not any(str(key).startswith("faf/") for key in metrics) and not any(
            key in metrics for key in ("affine_oracle_dice", "effective_slot_number", "coverage_score")
        ):
            return {}
        return out

    @classmethod
    def _functional_anchor_metrics(cls, metrics: Mapping[str, Any]) -> dict[str, Any]:
        aliases = {
            "base_dice": ("base_dice", "base_only_dice_frame_mean", "functional_anchor/base_dice"),
            "anchor_only_dice": ("anchor_only_dice", "anchor_only_dice_frame_mean", "functional_anchor/anchor_only_dice"),
            "proposal_dice": ("proposal_dice", "proposal_dice_frame_mean", "functional_anchor/proposal_dice"),
            "final_dice": ("final_dice", "dice_frame_mean", "dice", "functional_anchor/final_dice"),
            "final_minus_base": ("final_minus_base", "functional_anchor/final_minus_base"),
            "final_minus_anchor": ("final_minus_anchor", "functional_anchor/final_minus_anchor"),
            "proposal_minus_anchor": ("proposal_minus_anchor", "functional_anchor/proposal_minus_anchor"),
            "residual_l1": ("residual_l1", "functional_anchor/residual_l1"),
            "residual_l2": ("residual_l2", "functional_anchor/residual_l2"),
            "residual_boundary_ratio": ("residual_boundary_ratio", "functional_anchor/residual_boundary_ratio"),
            "residual_abs_mean": ("residual_abs_mean", "functional_anchor/residual_abs_mean"),
            "residual_abs_max": ("residual_abs_max", "functional_anchor/residual_abs_max"),
            "residual_clip_hit_ratio": ("residual_clip_hit_ratio", "functional_anchor/residual_clip_hit_ratio"),
            "residual_scale": ("residual_scale", "functional_anchor/residual_scale"),
            "delta_abs_mean": ("delta_abs_mean", "functional_anchor/delta_abs_mean"),
            "base_logit_abs_mean": ("base_logit_abs_mean", "functional_anchor/base_logit_abs_mean"),
            "anchor_logit_abs_mean": ("anchor_logit_abs_mean", "functional_anchor/anchor_logit_abs_mean"),
            "proposal_logit_abs_mean": ("proposal_logit_abs_mean", "functional_anchor/proposal_logit_abs_mean"),
            "final_logit_abs_mean": ("final_logit_abs_mean", "functional_anchor/final_logit_abs_mean"),
            "base_logit_std": ("base_logit_std", "functional_anchor/base_logit_std"),
            "anchor_logit_std": ("anchor_logit_std", "functional_anchor/anchor_logit_std"),
            "proposal_logit_std": ("proposal_logit_std", "functional_anchor/proposal_logit_std"),
            "final_logit_std": ("final_logit_std", "functional_anchor/final_logit_std"),
            "base_prob_mean": ("base_prob_mean", "functional_anchor/base_prob_mean"),
            "anchor_prob_mean": ("anchor_prob_mean", "functional_anchor/anchor_prob_mean"),
            "proposal_prob_mean": ("proposal_prob_mean", "functional_anchor/proposal_prob_mean"),
            "final_prob_mean": ("final_prob_mean", "functional_anchor/final_prob_mean"),
            "anchor_temperature": ("anchor_temperature", "functional_anchor/anchor_temperature"),
            "shape_residual_norm": ("shape_residual_norm", "functional_anchor/shape_residual_norm"),
            "boundary_residual_norm": ("boundary_residual_norm", "functional_anchor/boundary_residual_norm"),
            "area_curve_smoothness": ("area_curve_smoothness", "functional_anchor/area_curve_smoothness"),
            "area_smoothness": ("area_smoothness", "functional_anchor/area_smoothness"),
            "area_acceleration": ("area_acceleration", "functional_anchor/area_acceleration"),
            "temporal_jitter": ("temporal_jitter", "temporal_drift", "functional_anchor/temporal_jitter"),
            "anchor_temporal_consistency": ("anchor_temporal_consistency", "functional_anchor/anchor_temporal_consistency"),
            "slot_entropy": ("slot_entropy", "functional_anchor/slot_entropy"),
            "ED_slot_usage": ("ed_slot_usage", "ED_slot_usage", "functional_anchor/ED_slot_usage"),
            "slot_usage_ed": ("slot_usage_ed", "functional_anchor/slot_usage_ed"),
            "slot_usage_early_systole": ("early_systole_slot_usage", "functional_anchor/slot_usage_early_systole"),
            "ES_slot_usage": ("es_slot_usage", "ES_slot_usage", "functional_anchor/ES_slot_usage"),
            "slot_usage_es": ("slot_usage_es", "functional_anchor/slot_usage_es"),
            "slot_usage_early_diastole": ("early_diastole_slot_usage", "functional_anchor/slot_usage_early_diastole"),
            "slot_usage_uncertain": ("uncertain_slot_usage", "functional_anchor/slot_usage_uncertain"),
            "slot_max_prob_mean": ("slot_max_prob", "slot_max_prob_mean", "functional_anchor/slot_max_prob_mean"),
            "slot_max_prob_std": ("slot_max_prob_std", "functional_anchor/slot_max_prob_std"),
            "slot_area_order_violation": ("slot_area_order_violation", "functional_anchor/slot_area_order_violation"),
            "slot_order_loss": ("slot_order_loss", "functional_anchor/slot_order_loss"),
            "slot_area_ed": ("slot_area_ed", "functional_anchor/slot_area_ed"),
            "slot_area_early_systole": ("slot_area_early_systole", "functional_anchor/slot_area_early_systole"),
            "slot_area_es": ("slot_area_es", "functional_anchor/slot_area_es"),
            "slot_area_early_diastole": ("slot_area_early_diastole", "functional_anchor/slot_area_early_diastole"),
            "slot_area_uncertain": ("slot_area_uncertain", "functional_anchor/slot_area_uncertain"),
            "phase_source": ("phase_source", "functional_anchor/phase_source"),
            "phase_source_metadata_ratio": ("phase_source_metadata_ratio", "functional_anchor/phase_source_metadata_ratio"),
            "phase_source_area_ratio": ("phase_source_area_ratio", "functional_anchor/phase_source_area_ratio"),
            "phase_source_time_ratio": ("phase_source_time_ratio", "functional_anchor/phase_source_time_ratio"),
            "phase_loss": ("phase_loss", "aux_functional_anchor_phase_consistency", "functional_anchor/phase_loss"),
            "phase_reliability": ("phase_reliability", "functional_anchor/phase_reliability"),
            "phase_reliability_mean": ("phase_reliability_mean", "functional_anchor/phase_reliability_mean"),
            "phase_reliability_std": ("phase_reliability_std", "functional_anchor/phase_reliability_std"),
            "phase_reliability_min": ("phase_reliability_min", "functional_anchor/phase_reliability_min"),
            "phase_reliability_low_ratio": ("phase_reliability_low_ratio", "functional_anchor/phase_reliability_low_ratio"),
            "state_norm": ("state_norm", "functional_anchor/state_norm"),
            "state_delta_norm": ("state_delta_norm", "functional_anchor/state_delta_norm"),
            "state_update_norm": ("state_update_norm", "functional_anchor/state_update_norm"),
            "state_delta_ratio": ("state_delta_ratio", "functional_anchor/state_delta_ratio"),
            "ode_raw_delta_norm": ("ode_raw_delta_norm", "functional_anchor/ode_raw_delta_norm"),
            "ode_update_norm": ("ode_update_norm", "functional_anchor/ode_update_norm"),
            "ode_clamp_ratio": ("ode_clamp_ratio", "functional_anchor/ode_clamp_ratio"),
            "gate_mean_low": ("gate_mean_low", "functional_anchor/gate_mean_low"),
            "gate_mean_mid": ("gate_mean_mid", "functional_anchor/gate_mean_mid"),
            "gate_mean_high": ("gate_mean_high", "functional_anchor/gate_mean_high"),
            "inject_gate_low": ("inject_gate_low", "functional_anchor/inject_gate_low"),
            "inject_gate_mid": ("inject_gate_mid", "functional_anchor/inject_gate_mid"),
            "inject_gate_high": ("inject_gate_high", "functional_anchor/inject_gate_high"),
            "inject_gate_dec": ("inject_gate_dec", "functional_anchor/inject_gate_dec"),
            "confidence_mean": ("confidence_mean", "functional_anchor/confidence_mean"),
            "confidence_std": ("confidence_std", "functional_anchor/confidence_std"),
            "trust_mean": ("trust_mean", "functional_anchor/trust_mean"),
            "trust_std": ("trust_std", "functional_anchor/trust_std"),
            "trust_spatial_std": ("trust_spatial_std", "functional_anchor/trust_spatial_std"),
            "trust_temporal_std": ("trust_temporal_std", "functional_anchor/trust_temporal_std"),
            "trust_disagreement_corr": ("trust_disagreement_corr", "functional_anchor/trust_disagreement_corr"),
            "anchor_trust_ratio": ("anchor_trust_ratio", "functional_anchor/anchor_trust_ratio"),
            "image_trust_ratio": ("image_trust_ratio", "functional_anchor/image_trust_ratio"),
            "base_area_range": ("base_area_range", "functional_anchor/base_area_range"),
            "base_area_std": ("base_area_std", "functional_anchor/base_area_std"),
            "anchor_area_range": ("anchor_area_range", "functional_anchor/anchor_area_range"),
            "anchor_area_std": ("anchor_area_std", "functional_anchor/anchor_area_std"),
            "proposal_area_range": ("proposal_area_range", "functional_anchor/proposal_area_range"),
            "proposal_area_std": ("proposal_area_std", "functional_anchor/proposal_area_std"),
            "final_area_range": ("final_area_range", "functional_anchor/final_area_range"),
            "final_area_std": ("final_area_std", "functional_anchor/final_area_std"),
            "ED_ES_area_gap": ("ED_ES_area_gap", "functional_anchor/ED_ES_area_gap"),
            "ED_ES_area_ratio": ("ED_ES_area_ratio", "functional_anchor/ED_ES_area_ratio"),
        }
        out = {}
        for dst, keys in aliases.items():
            for key in keys:
                if key in metrics:
                    out[dst] = metrics[key]
                    break
        if "final_minus_base" not in out and "final_dice" in out and "base_dice" in out:
            final = cls._to_float(out["final_dice"])
            base = cls._to_float(out["base_dice"])
            if final is not None and base is not None:
                out["final_minus_base"] = final - base
        if "final_minus_anchor" not in out and "final_dice" in out and "anchor_only_dice" in out:
            final = cls._to_float(out["final_dice"])
            anchor = cls._to_float(out["anchor_only_dice"])
            if final is not None and anchor is not None:
                out["final_minus_anchor"] = final - anchor
        if "proposal_minus_anchor" not in out and "proposal_dice" in out and "anchor_only_dice" in out:
            proposal = cls._to_float(out["proposal_dice"])
            anchor = cls._to_float(out["anchor_only_dice"])
            if proposal is not None and anchor is not None:
                out["proposal_minus_anchor"] = proposal - anchor
        if not any(str(key).startswith("functional_anchor/") for key in metrics) and not any(
            key in metrics for key in ("anchor_only_dice", "proposal_dice", "base_dice")
        ):
            return {}
        return out

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        if not self.artifacts_enabled:
            if self.artifacts_required:
                raise RuntimeError(f"MLflow artifact logging is disabled but required: {path}")
            return
        path = Path(path)
        if path.exists() and path.is_file():
            def _artifact():
                import mlflow

                mlflow.log_artifact(str(path), artifact_path=artifact_path)

            self._call(_artifact, "log_artifact", required=self.artifacts_required)

    def log_artifacts(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        if not self.artifacts_enabled:
            if self.artifacts_required:
                raise RuntimeError(f"MLflow artifact logging is disabled but required: {path}")
            return
        path = Path(path)
        if path.exists() and path.is_dir():
            def _artifacts():
                import mlflow

                mlflow.log_artifacts(str(path), artifact_path=artifact_path)

            self._call(_artifacts, "log_artifacts", required=self.artifacts_required)

    def log_checkpoint(self, path: str | Path, *, artifact_name: str | None = None, name: str | None = None) -> None:
        path = Path(path)
        artifact_name = artifact_name or name
        if artifact_name and path.name != artifact_name:
            with tempfile.TemporaryDirectory(prefix="mlflow_ckpt_") as tmp:
                tmp_path = Path(tmp) / artifact_name
                shutil.copy2(path, tmp_path)
                self.log_artifact(tmp_path, artifact_path="checkpoints")
            return
        self.log_artifact(path, artifact_path="checkpoints")

    def log_evaluation_result(
        self,
        result: Any,
        *,
        step: int | None = None,
        log_artifacts: bool = False,
    ) -> None:
        if not self.enabled or result is None:
            return
        mode = str(getattr(result, "mode", "eval"))
        summary_metrics = dict(getattr(result, "summary_metrics", {}) or {})
        self.log_eval_summary(summary_metrics, mode=mode, step=step)

        if not log_artifacts:
            return

        with tempfile.TemporaryDirectory(prefix="mlflow_eval_") as tmp:
            tmp_path = Path(tmp)
            summary_path = tmp_path / "summary.json"
            summary_path.write_text(json.dumps(summary_metrics, indent=2, sort_keys=True), encoding="utf-8")
            self.log_artifact(summary_path, artifact_path="eval")

            threshold_sweep = dict(getattr(result, "threshold_sweep", {}) or {})
            if threshold_sweep:
                sweep_path = tmp_path / "threshold_sweep.csv"
                self._write_csv(sweep_path, [{"threshold": key, "dice": value} for key, value in threshold_sweep.items()])
                self.log_artifact(sweep_path, artifact_path="eval")

            for attr, filename in (
                ("per_video_metrics", "per_video_metrics.csv"),
                ("per_frame_metrics", "per_frame_metrics.csv"),
            ):
                rows = list(getattr(result, attr, []) or [])
                if rows:
                    csv_path = tmp_path / filename
                    self._write_csv(csv_path, rows)
                    self.log_artifact(csv_path, artifact_path="eval")

        for artifact in getattr(result, "visual_artifacts", []) or []:
            self.log_artifact(artifact, artifact_path="visuals")

    def log_run_logs(self) -> None:
        if not self.enabled:
            return
        candidates = [self.run_dir / "train.log"]
        configured = self._cfg_get("command_log_path", None)
        if configured:
            candidates.append(Path(str(configured)).expanduser())
        seen = set()
        for path in candidates:
            path = Path(path)
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            self.log_artifact(path, artifact_path="logs")

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
        self._log_json_artifact(env, "runtime.json", artifact_path="env")
        self._log_text_artifact(self._run_command(["pip", "freeze"]), "pip_freeze.txt", artifact_path="env")
        self._log_text_artifact(self._run_command(["nvidia-smi"]), "nvidia_smi.txt", artifact_path="env")
        torch_info = {key: env[key] for key in env if key.startswith("torch") or key.startswith("cuda")}
        self._log_json_artifact(torch_info, "torch_info.json", artifact_path="env")

    def log_git_info(self) -> None:
        if not self.enabled:
            return
        root = Path(__file__).resolve().parents[1]

        def git(*args: str) -> str:
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

        info: dict[str, Any] = {}
        status = ""
        diff = ""
        try:
            info["commit"] = git("rev-parse", "HEAD")
            info["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
            status = git("status", "--short")
            diff = git("diff", "--")
            info["status_short"] = status
        except Exception as exc:
            info["error"] = str(exc)
        self._log_json_artifact(info, "git.json", artifact_path="source")
        self._log_text_artifact(status, "git_status.txt", artifact_path="source")
        self._log_text_artifact(diff, "git_diff.patch", artifact_path="source")

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

    def _log_text_artifact(self, payload: str, filename: str, *, artifact_path: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / filename
        path.write_text(payload or "", encoding="utf-8")
        self.log_artifact(path, artifact_path=artifact_path)

    def _call(self, fn, action: str, *, required: bool) -> Any:
        try:
            with self._timeout(self.timeout_seconds):
                return fn()
        except Exception as exc:
            if required:
                raise RuntimeError(f"MLflow {action} failed: {exc}") from exc
            return None

    @staticmethod
    @contextmanager
    def _timeout(seconds: int):
        if seconds <= 0 or not hasattr(signal, "SIGALRM"):
            yield
            return
        previous = signal.getsignal(signal.SIGALRM)

        def _handler(signum, frame):
            raise TimeoutError(f"operation exceeded {seconds}s")

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    @staticmethod
    def _run_command(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
        except Exception as exc:
            return str(exc)

    @staticmethod
    def _standard_eval_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "dice": metrics.get("dice", metrics.get("dice_frame_mean")),
            "iou": metrics.get("iou", metrics.get("iou_frame_mean")),
            "hd95": metrics.get("hd95", metrics.get("hd95_original", metrics.get("hd95_resized"))),
            "assd": metrics.get("assd", metrics.get("assd_original", metrics.get("assd_resized"))),
            "phase/ED_Dice": metrics.get("ed_dice"),
            "phase/ES_Dice": metrics.get("es_dice"),
            "phase/ED_HD95": metrics.get("ed_hd95"),
            "phase/ES_HD95": metrics.get("es_hd95"),
            "overall/Dice": metrics.get("overall_dice", metrics.get("dice", metrics.get("dice_frame_mean"))),
            "overall/HD95": metrics.get("overall_hd95", metrics.get("hd95", metrics.get("hd95_original", metrics.get("hd95_resized")))),
            "area_smoothness": metrics.get("area_smoothness"),
            "area_acceleration": metrics.get("area_acceleration"),
            "temporal_jitter": metrics.get("temporal_jitter", metrics.get("temporal_drift")),
        }

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

    @classmethod
    def _extract_config_params(cls, value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
        root = prefix.split(".", 1)[0] if prefix else ""
        if prefix and root not in cls.PARAM_ROOTS:
            return {}
        if isinstance(value, Mapping):
            out = {}
            for key, item in value.items():
                next_key = f"{prefix}.{key}" if prefix else str(key)
                out.update(cls._extract_config_params(item, next_key))
            return out
        normalized = cls._param_value(value)
        if normalized is None:
            return {}
        return {prefix: normalized}

    @staticmethod
    def _param_value(value: Any) -> Any | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value if math.isfinite(float(value)) else None
        if isinstance(value, str):
            return value if len(value) <= 200 else None
        if isinstance(value, (list, tuple)) and len(value) <= 8:
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                text = ",".join(str(item) for item in value)
                return text if len(text) <= 200 else None
        return None

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
