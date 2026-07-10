from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data.distributed import DistributedSampler

from evaluation import EvaluationResult, Evaluator
from gdkvm_project.evaluation import align_logits_to_target, binary_dice_iou, collect_dpfr_diagnostics
from gdkvm_project.losses import LossComputer
from models.registry import build_model
from training.hooks import log_final_metrics, log_train_metrics
from training.logging import TrainingLogger
from training.parameter_groups import get_parameter_groups
from utils.frame_validity import build_default_endpoint_mask, normalize_frame_validity_mask

log = logging.getLogger(__name__)


def build_model_from_cfg(cfg: DictConfig, device: torch.device | str):
    return build_model(cfg, device=device)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, value in model_state.items():
            value = value.detach()
            if key not in self.state:
                self.state[key] = value.clone()
            elif torch.is_floating_point(value):
                self.state[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                self.state[key].copy_(value)

    def state_dict(self) -> dict:
        return {key: value.detach().clone() for key, value in self.state.items()}

    def load_state_dict(self, state: dict) -> None:
        self.state = {key: value.detach().clone() for key, value in state.items()}


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        stage_cfg: DictConfig,
        log: TrainingLogger,
        run_path: str,
        train_loader,
        val_loader,
        test_loader,
        mlflow_logger=None,
    ):
        self.cfg = cfg
        self.stage_cfg = stage_cfg
        self.log = log
        self.run_path = Path(run_path)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mlflow_logger = mlflow_logger
        self.evaluator = Evaluator(self)

        self.exp_id = str(cfg.get("exp_id", "experiment"))
        self.model_name = str(cfg.get("model_name", cfg.get("model", {}).get("name", "gdkvm")))
        self.stage = str(stage_cfg.get("name", "main_training"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        self.use_amp = bool(stage_cfg.get("amp", False)) and self.device.type == "cuda"
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.main_process = self.rank == 0

        model = build_model_from_cfg(cfg, self.device)
        if self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        else:
            self.model = model

        self.optimizer = optim.AdamW(
            get_parameter_groups(self.model, stage_cfg, print_log=self.main_process and bool(cfg.get("debug", False))),
            lr=float(stage_cfg.get("learning_rate", 1.0e-4)),
            weight_decay=float(stage_cfg.get("weight_decay", 0.0)),
            eps=1.0e-6 if self.use_amp else 1.0e-8,
            foreach=True,
        )
        self.loss_computer = LossComputer(cfg, stage_cfg)
        self.scaler = torch.amp.GradScaler(self.device.type, init_scale=8192, enabled=self.use_amp)
        self.clip_grad_norm = float(stage_cfg.get("clip_grad_norm", 0.0))
        self.scheduler = self._build_scheduler(stage_cfg)
        self.log_text_interval = int(cfg.get("log_text_interval", 100))
        self.log_image_interval = int(cfg.get("log_image_interval", 500))
        self.best_val_threshold = float(cfg.get("evaluation", {}).get("default_threshold", 0.5))
        self.best_val_metric = -1.0
        self.ema = None
        self.ema_enabled = bool(stage_cfg.get("use_ema", False))
        self.ema_eval = bool(stage_cfg.get("ema_eval", False))
        self._is_train = True
        self._summary_rows: list[dict[str, Any]] = []

    @property
    def model_without_ddp(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _build_scheduler(self, stage_cfg: DictConfig):
        schedule = str(stage_cfg.get("lr_schedule", "step")).lower()
        if schedule == "step":
            return MultiStepLR(
                self.optimizer,
                milestones=[int(x) for x in stage_cfg.get("lr_schedule_steps", [])],
                gamma=float(stage_cfg.get("lr_schedule_gamma", 0.1)),
            )
        total = max(int(stage_cfg.get("num_iterations", 1)), 1)
        warmup = max(int(stage_cfg.get("lr_warmup_iters", 0)), 0)
        min_ratio = float(stage_cfg.get("lr_min_ratio", 0.0))

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return max(float(step + 1) / float(warmup), 1.0e-6)
            if schedule in {"cosine", "warmup_cosine"}:
                progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
                return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        return LambdaLR(self.optimizer, lr_lambda)

    def _move_to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = self._move_to_device(item)
            return value
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        return value

    def _phase_init(self, mode: str) -> str:
        return str(self.cfg.get("phase_init", {}).get(mode, self.cfg.get("evaluation", {}).get("init_mode", "pred_or_zero")))

    def _resolve_supervised_indices(self, data: dict) -> torch.Tensor:
        batch_size, num_frames = data["rgb"].shape[:2]
        mask = data.get("loss_visibility", data.get("label_valid"))
        if mask is None:
            return build_default_endpoint_mask(batch_size, num_frames, device=self.device)
        return normalize_frame_validity_mask(mask, batch_size=batch_size, total_frames=num_frames, device=self.device)

    def train(self):
        self._is_train = True
        self.model.train()
        return self

    def val(self):
        self._is_train = False
        self.model.eval()
        return self

    def do_pass(self, data, it: int = 0) -> float:
        self.train()
        self._move_to_device(data)
        data["init_mode"] = self._phase_init("train")
        data["current_iter"] = int(it)
        data["global_step"] = int(it)
        data["current_epoch"] = int(it) // max(len(self.train_loader), 1)
        data["iters_per_epoch"] = max(len(self.train_loader), 1)

        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
            out = self.model(data)
            data.update(out)
            supervised_indices = self._resolve_supervised_indices(data)
            data["supervised_indices"] = supervised_indices
            num_objects = data.get("num_objects", out.get("num_objects", [1] * data["rgb"].shape[0]))
            losses = self.loss_computer.compute(data, num_objects)
            loss = losses["total_loss"]

        if not torch.isfinite(loss):
            if self.main_process:
                self.log.warning("[Trainer] non-finite loss at iter %s; skipping batch", it)
            return 0.0

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.clip_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model_without_ddp)

        loss_value = float(loss.detach().item())
        if self.main_process and (it % self.log_text_interval == 0):
            self.log.log_scalar("loss", loss_value, it)
            log_train_metrics(self, losses, loss, it)
        return loss_value

    def evaluate(self, val_loader, epoch, run_path, it, local_rank=None, world_size=None, full_eval: bool = False):
        result = self.evaluator.evaluate(val_loader, "val", epoch, run_path, it, full_eval=full_eval)
        self._log_eval_result(result, it, full_eval)
        self._maybe_save_best(result.summary_metrics, epoch, it)
        return result.summary_metrics

    def test(self, test_loader, epoch, run_path, it, local_rank=None, world_size=None, full_eval: bool = False):
        result = self.evaluator.evaluate(test_loader, "test", epoch, run_path, it, full_eval=full_eval)
        self._log_eval_result(result, it, full_eval)
        return result.summary_metrics

    def _log_eval_result(self, result: EvaluationResult, it: int, full_eval: bool) -> None:
        if not self.main_process:
            return
        log_final_metrics(self, result.summary_metrics, result.mode, it, result.epoch)
        logger = getattr(self, "mlflow_logger", None)
        if logger is not None and hasattr(logger, "log_evaluation_result"):
            logger.log_evaluation_result(result, step=it, log_artifacts=full_eval)

    def _forward_eval(self, batch: dict) -> dict:
        out = self.model(batch)
        batch.update(out)
        return out

    def _run_evaluation_impl(self, data_loader, mode: str, epoch: int, run_path, it: int, *, full_eval: bool = False):
        if self.is_distributed:
            dist.barrier()
        if isinstance(getattr(data_loader, "sampler", None), DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        prev_train = self.model.training
        self.model.eval()
        threshold = float(self.best_val_threshold if mode == "test" else self.cfg.get("evaluation", {}).get("default_threshold", 0.5))
        totals: dict[str, float] = {
            "dice_frame_sum": 0.0,
            "dice_frame_count": 0.0,
            "iou_frame_sum": 0.0,
            "iou_frame_count": 0.0,
        }
        dpfr_rows: list[dict[str, float]] = []

        with torch.no_grad():
            for batch in data_loader:
                self._move_to_device(batch)
                batch["init_mode"] = self._phase_init(mode)
                supervised = self._resolve_supervised_indices(batch)
                batch["supervised_indices"] = supervised
                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    out = self._forward_eval(batch)
                logits = out.get("logits")
                if not torch.is_tensor(logits):
                    frame_logits = [out[f"logits_{ti}"] for ti in range(batch["rgb"].shape[1]) if f"logits_{ti}" in out]
                    if not frame_logits:
                        continue
                    logits = torch.stack(frame_logits, dim=1)
                gt = batch["cls_gt"]
                if gt.dim() == 5:
                    gt = gt.squeeze(2)
                logits = align_logits_to_target(logits, gt)
                if logits.shape[:2] != gt.shape[:2]:
                    raise ValueError(
                        f"Evaluation frame grain mismatch: logits={tuple(logits.shape)} target={tuple(gt.shape)}"
                    )
                pred = torch.softmax(logits.float(), dim=2)[:, :, 1] >= threshold
                target = gt.long() > 0
                for bi in range(pred.shape[0]):
                    frame_ids = torch.nonzero(supervised[bi], as_tuple=False).flatten().tolist()
                    for ti in frame_ids:
                        dice, iou = binary_dice_iou(pred[bi, ti], target[bi, ti])
                        totals["dice_frame_sum"] += dice
                        totals["dice_frame_count"] += 1.0
                        totals["iou_frame_sum"] += iou
                        totals["iou_frame_count"] += 1.0
                dpfr_rows.append(collect_dpfr_diagnostics(batch, out, supervised))

        if prev_train:
            self.model.train()
        metrics = {
            "dice_frame_mean": totals["dice_frame_sum"] / max(totals["dice_frame_count"], 1.0),
            "iou_frame_mean": totals["iou_frame_sum"] / max(totals["iou_frame_count"], 1.0),
            "best_val_threshold": threshold,
            "frame_count": totals["dice_frame_count"],
            "metric_space": "target_mask_size",
            "eval_height": int(batch["cls_gt"].shape[-2]) if "batch" in locals() else 0,
            "eval_width": int(batch["cls_gt"].shape[-1]) if "batch" in locals() else 0,
        }
        metrics["dice"] = metrics["dice_frame_mean"]
        metrics["iou"] = metrics["iou_frame_mean"]
        metrics.update(_mean_metric_dicts(dpfr_rows))
        self._record_summary_row(mode, epoch, it, metrics)
        return EvaluationResult(mode=mode, iteration=it, epoch=epoch, summary_metrics=metrics)

    def _record_summary_row(self, mode: str, epoch: int, it: int, metrics: dict[str, float]) -> None:
        if not self.main_process:
            return
        row = {"mode": mode, "epoch": int(epoch), "iteration": int(it), **metrics}
        self._summary_rows.append(row)
        summary_path = self.run_path / "summary.json"
        summary_path.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        csv_path = self.run_path / "summary.csv"
        keys = sorted({key for item in self._summary_rows for key in item})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._summary_rows)

    def _maybe_save_best(self, metrics: dict[str, float], epoch: int, it: int) -> None:
        score = float(metrics.get("dice_frame_mean", metrics.get("dice", 0.0)))
        if not self.main_process or score <= self.best_val_metric:
            return
        self.best_val_metric = score
        self.best_val_threshold = float(metrics.get("best_val_threshold", self.best_val_threshold))
        self.run_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_without_ddp.state_dict(), self.run_path / "best_raw.pth")
        payload = {
            "iteration": int(it),
            "epoch": int(epoch),
            "metric_name": "dice_frame_mean",
            "metric": score,
            "best_val_threshold": self.best_val_threshold,
            "model_name": self.model_name,
        }
        (self.run_path / "best_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def save_weights(self, it: int):
        if not self.main_process:
            return
        self.run_path.mkdir(parents=True, exist_ok=True)
        path = self.run_path / f"{self.model_name}_iter_{it}.pth"
        torch.save(self.model_without_ddp.state_dict(), path)
        torch.save(self.model_without_ddp.state_dict(), self.run_path / "latest_weights.pth")
        self.log.info("Saved weights: %s", path)

    def save_checkpoint(self, it: int):
        if not self.main_process:
            return
        self.run_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "it": int(it),
            "model": self.model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_threshold": self.best_val_threshold,
            "best_val_metric": self.best_val_metric,
        }
        path = self.run_path / f"{self.model_name}_{self.stage}_ckpt_{it}.pth"
        torch.save(payload, path)
        torch.save(payload, self.run_path / "latest_checkpoint.pth")
        self.log.info("Saved checkpoint: %s", path)

    def upload_summary_artifact(self) -> None:
        if not self.main_process:
            return
        logger = getattr(self, "mlflow_logger", None)
        summary_path = self.run_path / "summary.csv"
        if logger is not None and summary_path.exists() and hasattr(logger, "log_artifact"):
            logger.log_artifact(summary_path, artifact_path="summary")


def _mean_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if value is None or not math.isfinite(float(value)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0.0) + 1.0
    return {key: sums[key] / max(counts[key], 1.0) for key in sums}
