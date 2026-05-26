import os
import logging
import csv
import copy
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig

from evaluation import EvaluationResult, Evaluator
from training.logging import TrainingLogger
from utils.log_integrator import Integrator
from utils.time_estimator import TimeEstimator
from models.registry import build_model
from training.parameter_groups import get_parameter_groups
from losses import LossComputer
from visualization import visualize_sequence
from utils.frame_validity import (
    mask_to_frame_ids,
    normalize_frame_validity_mask,
    summarize_frame_mask,
)

from monai.metrics import (
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
)

log = logging.getLogger(__name__)


def build_model_from_cfg(cfg: DictConfig, device: torch.device | str):
    """Build a model through the public model registry."""
    return build_model(cfg, device=device)


def _contains_policy_head(name: str) -> bool:
    return (
        "prototype_manager.policy_head" in name
        or "memory_core.prototype_manager.policy_head" in name
    )


class ModelEMA:
    """Small state-dict EMA helper that works with plain modules and DDP modules."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, value in model_state.items():
            value = value.detach()
            if key not in self.state:
                self.state[key] = value.clone()
                continue
            if torch.is_floating_point(value):
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

        self.exp_id = cfg["exp_id"]
        self.model_name = str(cfg.get("model_name", cfg.model.get("name", "BanditPM")))
        self.stage = stage_cfg["name"]
        self.crop_size = stage_cfg["crop_size"]

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        self.use_amp = bool(stage_cfg.amp) and self.device.type == "cuda"

        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.main_process = self.rank == 0

        model = build_model_from_cfg(cfg, self.device)
        model = model.to(memory_format=torch.channels_last)
        self._apply_training_freeze(model)
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.model = model

        if self.main_process:
            try:
                param_count = sum(p.nelement() for p in self.model.parameters()) / 1e6
                self.log.info(f"Model Parameters: {param_count:.2f}M")
                self._log_metrics({"model/parameters_m": param_count}, step=0)
            except Exception:
                self.log.info("Model Parameters: Count failed")

        self.train_integrator = Integrator(self.log, distributed=True)
        self._is_train = True

        parameter_groups = get_parameter_groups(
            self.model, stage_cfg, print_log=self.main_process
        )
        self.optimizer = optim.AdamW(
            parameter_groups,
            lr=stage_cfg["learning_rate"],
            weight_decay=stage_cfg["weight_decay"],
            eps=1e-6 if self.use_amp else 1e-8,
            foreach=True,
        )
        self.loss_computer = LossComputer(cfg, stage_cfg)
        self.scaler = torch.amp.GradScaler(
            self.device.type, init_scale=8192, enabled=self.use_amp
        )
        self.clip_grad_norm = stage_cfg["clip_grad_norm"]

        self._init_scheduler(stage_cfg)

        self.log_text_interval = cfg.get("log_text_interval", 100)
        self.log_image_interval = cfg.get("log_image_interval", 500)
        if cfg.get("debug", False):
            self.log_text_interval = self.log_image_interval = 1

        self.log.time_estimator = TimeEstimator(
            stage_cfg.get("num_iterations", 3000), self.log_text_interval
        )

        self._init_metrics()
        self.commit_hash = self._resolve_commit_hash()
        self.best_val_threshold = float(
            self.cfg.get("evaluation", {}).get("default_threshold", 0.5)
        )
        self._best_val_threshold_ready = False
        self.ema = None
        self.ema_enabled = bool(stage_cfg.get("use_ema", False))
        self.ema_eval = bool(stage_cfg.get("ema_eval", self.ema_enabled))
        self.ema_start_iter = int(stage_cfg.get("ema_start_iter", 0))
        if self.ema_enabled:
            self.ema = ModelEMA(self.model_without_ddp, decay=float(stage_cfg.get("ema_decay", 0.999)))
            if self.main_process:
                self.log.info(
                    f"EMA enabled: decay={self.ema.decay} start_iter={self.ema_start_iter} eval={self.ema_eval}"
                )
        self.best_val_metric = -float("inf")

    @property
    def model_without_ddp(self) -> nn.Module:
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _apply_training_freeze(self, model: nn.Module) -> None:
        temporal_cfg = self.cfg.model.get("temporal_memory", {})
        bpm_cfg = temporal_cfg.get("bpm", {})
        freeze_backbone = bool(bpm_cfg.get("FREEZE_BACKBONE", False))
        train_policy_only = bool(bpm_cfg.get("TRAIN_POLICY_ONLY", False))

        if not freeze_backbone and not train_policy_only:
            return

        for _, param in model.named_parameters():
            param.requires_grad = True

        if freeze_backbone:
            for name, param in model.named_parameters():
                if name.startswith("image_encoder.") or name.startswith("mask_encoder."):
                    param.requires_grad = False

        if train_policy_only:
            for _, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if _contains_policy_head(name):
                    param.requires_grad = True

    def _init_scheduler(self, stage_cfg):
        if stage_cfg["lr_schedule"] == "constant":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda _: 1.0
            )
        elif stage_cfg["lr_schedule"] == "poly":
            total_num_iter = stage_cfg["num_iterations"]
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda x: (1 - (x / total_num_iter)) ** 0.9
            )
        elif stage_cfg["lr_schedule"] == "step":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                stage_cfg["lr_schedule_steps"],
                stage_cfg["lr_schedule_gamma"],
            )
        else:
            raise NotImplementedError(
                f"Scheduler {stage_cfg['lr_schedule']} not implemented"
            )

    def _init_metrics(self):
        self.conf_metric_names = [
            "precision",
            "recall",
            "accuracy",
            "specificity",
            "f1 score",
        ]
        self.conf_metric = ConfusionMatrixMetric(include_background=False, metric_name=self.conf_metric_names, reduction="mean")

    def _log_metrics(self, metrics: dict, *, step: int | None = None, prefix: str | None = None) -> None:
        logger = getattr(self, "mlflow_logger", None)
        if getattr(self, "main_process", False) and logger is not None:
            logger.log_metrics(metrics, step=step, prefix=prefix)

    def _resolve_commit_hash(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
            ).strip()
        except Exception:
            return "unknown"

    def _resolve_phase_init(self, phase: str) -> str:
        phase_cfg = self.cfg.get("phase_init", {})
        model_cfg = self.cfg.get("model", {})
        allow_oracle_init = bool(
            model_cfg.get(
                "allow_oracle_init_when_requested",
                model_cfg.get("use_first_frame_gt_init", True),
            )
        )
        default_mode = "oracle_gt" if allow_oracle_init else "pred_or_zero"
        return str(phase_cfg.get(phase, self.cfg.get("evaluation", {}).get("init_mode", default_mode)))

    def _oracle_init_allowed(self) -> bool:
        model_cfg = self.cfg.get("model", {})
        return bool(
            model_cfg.get(
                "allow_oracle_init_when_requested",
                model_cfg.get("use_first_frame_gt_init", True),
            )
        )

    def _resolve_eval_indices(self, data):
        T = data["rgb"].shape[1]
        eval_cfg = self.cfg.get("evaluation", {})
        frame_scope = str(eval_cfg.get("frame_scope", "supervised_only"))
        if frame_scope == "all_available":
            eval_valid = data.get("eval_valid")
            if eval_valid is not None:
                source = eval_valid
            else:
                source = data.get("label_valid")
        else:
            source = data.get("label_valid")

        if source is None:
            frame_mask = torch.ones((data["rgb"].shape[0], T), device=self.device, dtype=torch.bool)
            return self._apply_eval_exclusions(frame_mask)

        frame_mask = self._resolve_frame_valid_mask(source, batch_size=data["rgb"].shape[0], total_frames=T)
        if not frame_mask.any():
            frame_mask = self._resolve_supervised_indices(data)
        return self._apply_eval_exclusions(frame_mask)

    def _apply_eval_exclusions(self, frame_mask: torch.Tensor) -> torch.Tensor:
        eval_cfg = self.cfg.get("evaluation", {})
        if not bool(eval_cfg.get("exclude_init_frame", False)):
            return frame_mask
        frame_mask = frame_mask.clone()
        init_idx = int(eval_cfg.get("init_frame_index", 0))
        if 0 <= init_idx < frame_mask.shape[1]:
            frame_mask[:, init_idx] = False
        return frame_mask

    def _binary_overlap_metrics(self, pred: torch.Tensor, gt: torch.Tensor):
        pred = pred.float()
        gt = gt.float()
        inter = float((pred * gt).sum().item())
        pred_sum = float(pred.sum().item())
        gt_sum = float(gt.sum().item())
        union = pred_sum + gt_sum - inter
        if pred_sum == 0.0 and gt_sum == 0.0:
            return 1.0, 1.0
        dice = (2.0 * inter) / max(pred_sum + gt_sum, 1e-6)
        iou = inter / max(union, 1e-6)
        return dice, iou

    def _postprocess_enabled(self) -> bool:
        eval_cfg = self.cfg.get("evaluation", {})
        post_cfg = eval_cfg.get("postprocess", {})
        if isinstance(post_cfg, dict) or hasattr(post_cfg, "get"):
            default = str(self.cfg.get("model", {}).get("name", "")).lower() in {"anchor_ode_v2", "unext_anchor_ode_affine"}
            return bool(post_cfg.get("enabled", default))
        return bool(post_cfg)

    def _postprocess_binary_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not self._postprocess_enabled():
            return mask
        try:
            from scipy import ndimage
        except Exception:
            return mask

        eval_cfg = self.cfg.get("evaluation", {})
        post_cfg = eval_cfg.get("postprocess", {})
        min_size = int(post_cfg.get("min_size", 16)) if hasattr(post_cfg, "get") else 16
        keep_largest = bool(post_cfg.get("largest_component", True)) if hasattr(post_cfg, "get") else True
        fill_holes = bool(post_cfg.get("fill_holes", True)) if hasattr(post_cfg, "get") else True
        remove_small = bool(post_cfg.get("remove_small_objects", True)) if hasattr(post_cfg, "get") else True
        binary_closing = bool(post_cfg.get("binary_closing", True)) if hasattr(post_cfg, "get") else True
        structure = np.ones((3, 3), dtype=bool)
        device = mask.device
        dtype = mask.dtype
        arr = mask.detach().cpu().numpy().astype(bool)
        out = np.zeros_like(arr, dtype=np.float32)
        flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        flat_out = out.reshape(-1, out.shape[-2], out.shape[-1])
        for idx, item in enumerate(flat):
            if keep_largest:
                labels, num = ndimage.label(item, structure=structure)
                if num > 0:
                    counts = np.bincount(labels.ravel())
                    counts[0] = 0
                    largest = int(counts.argmax())
                    item = labels == largest
            if fill_holes:
                item = ndimage.binary_fill_holes(item)
            if remove_small and min_size > 1:
                labels, num = ndimage.label(item, structure=structure)
                if num > 0:
                    counts = np.bincount(labels.ravel())
                    keep = counts >= min_size
                    keep[0] = False
                    item = keep[labels]
            if binary_closing:
                item = ndimage.binary_closing(item, structure=structure)
            flat_out[idx] = item.astype(np.float32)
        return torch.as_tensor(out, device=device, dtype=dtype)

    def _surface_metrics_single(self, pred: torch.Tensor, gt: torch.Tensor):
        pred = pred.float()
        gt = gt.float()
        pred_sum = float(pred.sum().item())
        gt_sum = float(gt.sum().item())
        if pred_sum == 0.0 and gt_sum == 0.0:
            return 0.0, 0.0
        if pred_sum == 0.0 or gt_sum == 0.0:
            max_dim = float(max(pred.shape[-2], pred.shape[-1], gt.shape[-2], gt.shape[-1]))
            return max_dim, max_dim

        hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        assd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")
        hd_metric(y_pred=pred, y=gt)
        assd_metric(y_pred=pred, y=gt)
        hd95 = hd_metric.aggregate()
        assd = assd_metric.aggregate()
        hd95 = hd95.item() if isinstance(hd95, torch.Tensor) else float(hd95)
        assd = assd.item() if isinstance(assd, torch.Tensor) else float(assd)
        if not np.isfinite(hd95):
            hd95 = float(max(pred.shape[-2], pred.shape[-1]))
        if not np.isfinite(assd):
            assd = float(max(pred.shape[-2], pred.shape[-1]))
        return hd95, assd

    def _resize_to_original(self, pred: torch.Tensor, gt: torch.Tensor, original_hw):
        target_h = int(original_hw[0])
        target_w = int(original_hw[1])
        if pred.shape[-2:] == (target_h, target_w) and gt.shape[-2:] == (target_h, target_w):
            return pred, gt
        pred_up = F.interpolate(pred.float(), size=(target_h, target_w), mode="nearest")
        gt_up = F.interpolate(gt.float(), size=(target_h, target_w), mode="nearest")
        return pred_up, gt_up

    def _build_summary_row(self, mode: str, metrics: dict, epoch: int, it: int):
        metric_space = str(self.cfg.get("evaluation", {}).get("metric_space", "original"))
        init_mode = self._resolve_phase_init(mode)
        uses_oracle_gt = init_mode == "oracle_gt" and self._oracle_init_allowed()
        return {
            "mode": mode,
            "iteration": it,
            "epoch": epoch,
            "experiment_name": self.exp_id,
            "dataset": str(self.cfg.get("dataset_name", "")),
            "protocol_name": str(self.cfg.get("data", {}).get("protocol_name", "unknown")),
            "init_mode": init_mode,
            "oracle_gt_init_allowed": self._oracle_init_allowed(),
            "uses_oracle_gt": uses_oracle_gt,
            "frame_scope": str(self.cfg.get("evaluation", {}).get("frame_scope", "supervised_only")),
            "exclude_init_frame": bool(self.cfg.get("evaluation", {}).get("exclude_init_frame", False)),
            "init_frame_index": int(self.cfg.get("evaluation", {}).get("init_frame_index", 0)),
            "postprocess_enabled": self._postprocess_enabled(),
            "ema_enabled": getattr(self, "ema_enabled", False),
            "ema_eval": getattr(self, "ema_eval", False),
            "tta_enabled": self._tta_enabled(),
            "tta_modes": ",".join(self._tta_modes()),
            "metric_space": metric_space,
            "dice_frame_mean": metrics.get("dice_frame_mean", 0.0),
            "dice_video_mean": metrics.get("dice_video_mean", 0.0),
            "iou_frame_mean": metrics.get("iou_frame_mean", 0.0),
            "iou_video_mean": metrics.get("iou_video_mean", 0.0),
            "hd95_resized": metrics.get("hd95_resized", 0.0),
            "hd95_original": metrics.get("hd95_original", 0.0),
            "ed_dice": metrics.get("ed_dice", 0.0),
            "es_dice": metrics.get("es_dice", 0.0),
            "ed_hd95": metrics.get("ed_hd95", 0.0),
            "es_hd95": metrics.get("es_hd95", 0.0),
            "overall_dice": metrics.get("overall_dice", metrics.get("dice", 0.0)),
            "overall_hd95": metrics.get("overall_hd95", metrics.get("hd95", 0.0)),
            "assd_resized": metrics.get("assd_resized", 0.0),
            "assd_original": metrics.get("assd_original", 0.0),
            "temporal_drift": metrics.get("temporal_drift", 0.0),
            "temporal_dice_consistency": metrics.get("temporal_dice_consistency", 0.0),
            "area_smoothness": metrics.get("area_smoothness", 0.0),
            "centroid_jitter": metrics.get("centroid_jitter", 0.0),
            "threshold_0p5_dice_frame_mean": metrics.get("threshold_0p5_dice_frame_mean", metrics.get("dice_frame_mean", 0.0)),
            "best_val_threshold": metrics.get("best_val_threshold", getattr(self, "best_val_threshold", 0.5)),
            "best_threshold_dice_frame_mean": metrics.get("best_threshold_dice_frame_mean", metrics.get("dice_frame_mean", 0.0)),
            "teacher_forcing_prob": metrics.get("teacher_forcing_prob", 0.0),
            "gate_mean": metrics.get("gate_mean", 0.0),
            "residual_abs_mean": metrics.get("residual_abs_mean", 0.0),
            "memory_update_rate": metrics.get("memory_update_rate", 0.0),
            "base_only_dice_frame_mean": metrics.get("base_only_dice_frame_mean", 0.0),
            "guided_only_dice_frame_mean": metrics.get("guided_only_dice_frame_mean", 0.0),
            "prior_only_dice_frame_mean": metrics.get("prior_only_dice_frame_mean", 0.0),
            "best_ckpt_rule": str(self.cfg.get("evaluation", {}).get("best_ckpt_rule", "max_eval_dice_observed_no_reload")),
            "seed": int(self.cfg.get("seed", 42)),
            "commit_hash": self.commit_hash,
        }

    def _append_summary_row(self, row: dict):
        if not self.main_process or not bool(self.cfg.get("evaluation", {}).get("save_summary", True)):
            return
        summary_path = self.run_path / "summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        write_header = not summary_path.exists()
        with summary_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _move_to_device(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 4:
                    batch[key] = value.to(
                        self.device,
                        memory_format=torch.channels_last,
                        non_blocking=True,
                    )
                elif value.ndim == 5:
                    batch[key] = value.to(
                        self.device,
                        memory_format=torch.channels_last_3d,
                        non_blocking=True,
                    )
                else:
                    batch[key] = value.to(self.device, non_blocking=True)
        return batch

    def _ensure_finite_outputs(self, outputs):
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    outputs[key] = torch.nan_to_num(
                        value, nan=0.0, posinf=1e4, neginf=-1e4
                    )
        return outputs

    def _resolve_frame_valid_mask(self, source, batch_size: int, total_frames: int):
        return normalize_frame_validity_mask(
            source,
            batch_size=batch_size,
            total_frames=total_frames,
            device=self.device,
        )

    def _format_frame_mask(self, frame_mask: torch.Tensor, max_samples: int = 3):
        return summarize_frame_mask(frame_mask, max_samples=max_samples)

    def _mask_to_frame_ids(self, frame_mask: torch.Tensor) -> list[int]:
        return mask_to_frame_ids(frame_mask)

    def _resolve_supervised_indices(self, data):
        T = data["rgb"].shape[1]
        frame_mask = self._resolve_frame_valid_mask(
            data.get("label_valid"),
            batch_size=data["rgb"].shape[0],
            total_frames=T,
        )
        if not frame_mask.any(dim=1).all():
            raise ValueError("label_valid selects no supervised frames for at least one sample")
        return frame_mask

    def train(self):
        self._is_train = True
        self.model.train()
        return self

    def val(self):
        self._is_train = False
        self.model.eval()
        return self

    def do_pass(self, data, it=0):
        torch.set_grad_enabled(self._is_train)
        self._move_to_device(data)
        data["init_mode"] = self._resolve_phase_init("train")
        data["current_iter"] = it
        data["global_step"] = it
        data["current_epoch"] = it // max(len(self.train_loader), 1)
        data["iters_per_epoch"] = max(len(self.train_loader), 1)

        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
            out = self.model(data)
            out = self._ensure_finite_outputs(out)

            num_objects = out.get("num_objects", [1] * data["rgb"].shape[0])
            data.update(out)

            supervised_indices = self._resolve_supervised_indices(data)
            required_frame_ids = sorted(torch.nonzero(supervised_indices.any(dim=0), as_tuple=False).flatten().tolist())
            all_logits_keys = [f"logits_{ti}" for ti in required_frame_ids]
            if not all(k in data for k in all_logits_keys):
                raise KeyError(
                    f"Missing logits keys. Expected {all_logits_keys}, found {list(data.keys())}"
                )
            data.update(
                {
                    "supervised_indices": supervised_indices,
                }
            )

            losses = self.loss_computer.compute(data, num_objects)
            loss = losses["total_loss"]

        if not torch.isfinite(loss):
            if self.main_process:
                self.log.warning(
                    f"[Trainer] Loss is NaN/Inf at iter {it}, skipping batch."
                )
            return torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()

        if self.clip_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.ema is not None and (it + 1) >= self.ema_start_iter:
            self.ema.update(self.model_without_ddp)

        if it % self.log_text_interval == 0:
            loss_val = loss.item()
            self.log.log_scalar("loss", loss_val, it)

            if self.main_process:
                self._log_train_metrics(losses, loss_val, it)
                if it != 0:
                    self.log.log_scalar("lr", self.scheduler.get_last_lr()[0], it)
                self._log_bpm_stats(data, it)
                self._log_dynakey_stats(data, it)
                self._log_anchor_ode_stats(data, it)
                self._log_functional_anchor_stats(data, it)
                self._log_faf_stats(data, it)
                self._log_functional_anchor_grad_stats(it)

        return loss.detach()

    def _log_train_metrics(self, losses, total_loss, it):
        try:
            log_dict = {
                "total_loss": total_loss,
                "lr": self.scheduler.get_last_lr()[0],
            }
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    log_dict[k] = v.item()
            for attr, name in (
                ("lambda_functional_anchor_anchor", "anchor"),
                ("lambda_functional_anchor_base", "base"),
                ("lambda_functional_anchor_residual_l1", "residual_l1"),
                ("lambda_functional_anchor_boundary", "boundary_residual"),
                ("lambda_functional_anchor_phase", "phase_consistency"),
                ("lambda_functional_anchor_temp", "anchor_temporal"),
                ("lambda_functional_anchor_slot_order", "slot_area_order"),
                ("lambda_functional_anchor_phase_slot", "phase_slot_correlation"),
                ("lambda_functional_anchor_trust_l1", "trust_l1"),
                ("lambda_functional_anchor_trust_entropy", "trust_entropy"),
                ("lambda_functional_anchor_ode_raw_delta", "ode_raw_delta"),
                ("lambda_faf_anchor", "anchor"),
                ("lambda_faf_base", "base"),
                ("lambda_faf_coverage", "coverage"),
                ("lambda_faf_sparse", "sparse"),
                ("lambda_faf_diversity", "diversity"),
                ("lambda_faf_temporal", "temporal"),
                ("lambda_faf_write", "write"),
                ("lambda_faf_residual_smallness", "residual_smallness"),
                ("lambda_faf_affine", "affine"),
                ("lambda_faf_velocity", "velocity"),
            ):
                if hasattr(self.loss_computer, attr):
                    prefix = "lambda_faf" if attr.startswith("lambda_faf_") else "lambda_functional_anchor"
                    log_dict[f"{prefix}_{name}"] = getattr(self.loss_computer, attr)
            for group in self.optimizer.param_groups:
                name = group.get("name")
                if name == "functional_anchor_residual_heads":
                    log_dict["residual_head_lr"] = group.get("lr", 0.0)
            if str(self.cfg.get("model", {}).get("name", "")).lower() == "functional_anchor":
                model = self.model_without_ddp
                if hasattr(model, "_anchor_temperature"):
                    temp = model._anchor_temperature(self.device, torch.float32)
                    log_dict["anchor_temperature"] = float(temp.detach().item())
                if hasattr(model, "_residual_scale_at"):
                    scale = model._residual_scale_at({"current_iter": it}, self.device, torch.float32)
                    log_dict["residual_scale"] = float(scale.detach().item())
            logger = getattr(self, "mlflow_logger", None)
            if logger is not None:
                logger.log_train_step(log_dict, step=it)
        except Exception:
            pass

    def _grad_norm_for_prefixes(self, prefixes: tuple[str, ...]) -> float | None:
        total = 0.0
        found = False
        for name, param in self.model_without_ddp.named_parameters():
            if not any(name.startswith(prefix) for prefix in prefixes):
                continue
            if param.grad is None:
                continue
            found = True
            total += float(param.grad.detach().float().pow(2).sum().item())
        if not found:
            return None
        return total ** 0.5

    def _log_functional_anchor_grad_stats(self, it: int) -> None:
        if str(self.cfg.get("model", {}).get("name", "")).lower() != "functional_anchor":
            return
        metrics = {}
        for key, prefixes in {
            "grad/residual_head": ("residual_heads.",),
            "grad/shape_residual_head": ("residual_heads.shape_head.",),
            "grad/boundary_residual_head": ("residual_heads.boundary_head.",),
            "grad/trust_head": ("residual_heads.confidence_head.",),
            "grad/anchor_decoder": ("anchor_decoder.",),
            "grad/ode_state": ("state_ode.",),
            "grad/slot_predictor": ("anchor_bank.selector.",),
        }.items():
            value = self._grad_norm_for_prefixes(prefixes)
            if value is not None:
                metrics[key] = value
        temp = getattr(self.model_without_ddp, "anchor_temperature_raw", None)
        if isinstance(temp, torch.nn.Parameter) and temp.grad is not None:
            metrics["grad/anchor_temperature"] = float(temp.grad.detach().abs().item())
        if metrics:
            self._log_metrics(metrics, step=it)

    def _log_bpm_stats(self, data, it: int) -> None:
        bpm_keys = sorted(k for k in data.keys() if k.startswith("bpm_aux_"))
        if not bpm_keys:
            return

        try:
            metrics = {}
            action_tensors = []
            entropy_tensors = []
            agreement_tensors = []
            occupancy_tensors = []
            age_tensors = []
            usage_tensors = []
            conf_tensors = []
            for key in bpm_keys:
                aux = data[key]
                if "policy_actions" in aux:
                    action_tensors.append(aux["policy_actions"].detach().flatten())
                if "entropy" in aux:
                    entropy_tensors.append(aux["entropy"].detach().flatten())
                if "action_agreement" in aux:
                    agreement_tensors.append(aux["action_agreement"].detach().flatten())
                if "occupancy_ratio" in aux:
                    occupancy_tensors.append(aux["occupancy_ratio"].detach().flatten())
                if "bank_age" in aux:
                    age_tensors.append(aux["bank_age"].detach().flatten())
                if "bank_usage" in aux:
                    usage_tensors.append(aux["bank_usage"].detach().flatten())
                if "bank_conf" in aux:
                    conf_tensors.append(aux["bank_conf"].detach().flatten())

            if action_tensors:
                actions = torch.cat(action_tensors, dim=0)
                hist = torch.bincount(actions, minlength=4).float()
                hist = hist / hist.sum().clamp_min(1.0)
                names = ["keep", "refine", "replace", "spawn"]
                for idx, name in enumerate(names):
                    value = hist[idx].item()
                    self.log.log_scalar(f"bpm/action_{name}", value, it)
                    metrics[f"action_{name}"] = value

            if age_tensors:
                metrics["age_mean"] = torch.cat(age_tensors).mean().item()
                self.log.log_scalar("bpm/age_mean", metrics["age_mean"], it)
            if usage_tensors:
                metrics["usage_mean"] = torch.cat(usage_tensors).mean().item()
                self.log.log_scalar("bpm/usage_mean", metrics["usage_mean"], it)
            if conf_tensors:
                metrics["conf_mean"] = torch.cat(conf_tensors).mean().item()
                self.log.log_scalar("bpm/conf_mean", metrics["conf_mean"], it)
            if entropy_tensors:
                metrics["policy_entropy"] = torch.cat(entropy_tensors).mean().item()
                self.log.log_scalar("bpm/policy_entropy", metrics["policy_entropy"], it)
            if agreement_tensors:
                metrics["rule_learned_agreement"] = torch.cat(agreement_tensors).mean().item()
                self.log.log_scalar("bpm/rule_learned_agreement", metrics["rule_learned_agreement"], it)
            if occupancy_tensors:
                metrics["occupancy_ratio"] = torch.cat(occupancy_tensors).mean().item()
                self.log.log_scalar("bpm/occupancy_ratio", metrics["occupancy_ratio"], it)
            self._log_metrics(metrics, step=it, prefix="bpm")
        except Exception:
            pass

    def _log_dynakey_stats(self, data, it: int) -> None:
        memory_keys = sorted(k for k in data.keys() if k.startswith("memory_aux_"))
        if not memory_keys:
            return

        try:
            metrics = {}
            occupancy_tensors = []
            active_count_tensors = []
            entropy_tensors = []
            fallback_tensors = []
            prediction_error_tensors = []
            residual_tensors = []
            action_hist_tensors = []
            action_count_tensors = []
            valid_q_tensors = []
            invalid_target_tensors = []
            gate_tensors = []
            residual_abs_tensors = []
            base_abs_tensors = []
            memory_update_tensors = []
            rejected_update_tensors = []
            mid_gate_tensors = []
            mid_contrib_tensors = []
            enhanced_diff_tensors = []
            spatial_entropy_tensors = []
            for key in memory_keys:
                aux = data[key]
                if isinstance(aux, dict):
                    if torch.is_tensor(aux.get("gate_mean")):
                        gate_tensors.append(aux["gate_mean"].float().detach().flatten())
                    if torch.is_tensor(aux.get("residual_abs_mean")):
                        residual_abs_tensors.append(aux["residual_abs_mean"].float().detach().flatten())
                    if torch.is_tensor(aux.get("base_logits_abs_mean")):
                        base_abs_tensors.append(aux["base_logits_abs_mean"].float().detach().flatten())
                    for update_key in ("mask_memory_update_rate", "spatial_memory_update_rate"):
                        if torch.is_tensor(aux.get(update_key)):
                            memory_update_tensors.append(aux[update_key].float().detach().flatten())
                    for reject_key in ("rejected_update_count", "spatial_memory_rejected_count"):
                        if torch.is_tensor(aux.get(reject_key)):
                            rejected_update_tensors.append(aux[reject_key].float().detach().flatten())
                    if torch.is_tensor(aux.get("mid_memory_gate_mean")):
                        mid_gate_tensors.append(aux["mid_memory_gate_mean"].float().detach().flatten())
                    if torch.is_tensor(aux.get("mid_memory_contribution_norm")):
                        mid_contrib_tensors.append(aux["mid_memory_contribution_norm"].float().detach().flatten())
                    if torch.is_tensor(aux.get("enhanced_feature_diff_norm")):
                        enhanced_diff_tensors.append(aux["enhanced_feature_diff_norm"].float().detach().flatten())
                    if torch.is_tensor(aux.get("spatial_memory_entropy")):
                        spatial_entropy_tensors.append(aux["spatial_memory_entropy"].float().detach().flatten())
                dynakey_aux = aux.get("dynakey_aux") if isinstance(aux, dict) else None
                if not dynakey_aux:
                    continue
                if "occupancy_ratio" in dynakey_aux:
                    occupancy_tensors.append(dynakey_aux["occupancy_ratio"].detach().flatten())
                if "active_key_count" in dynakey_aux:
                    active_count_tensors.append(dynakey_aux["active_key_count"].float().detach().flatten())
                if "retrieval_entropy" in dynakey_aux:
                    entropy_tensors.append(dynakey_aux["retrieval_entropy"].detach().flatten())
                if "used_identity_fallback" in dynakey_aux:
                    fallback_tensors.append(dynakey_aux["used_identity_fallback"].float().detach().flatten())
                if "prediction_error" in dynakey_aux:
                    prediction_error_tensors.append(dynakey_aux["prediction_error"].detach().flatten())
                if "residual_norm" in dynakey_aux:
                    residual_tensors.append(dynakey_aux["residual_norm"].detach().flatten())
                if "action_hist" in dynakey_aux:
                    action_hist_tensors.append(dynakey_aux["action_hist"].detach())
                if "action_counts" in dynakey_aux:
                    action_count_tensors.append(dynakey_aux["action_counts"].detach())
                if torch.is_tensor(dynakey_aux.get("valid_q_samples")):
                    valid_q_tensors.append(dynakey_aux["valid_q_samples"].float().detach().flatten())
                if torch.is_tensor(dynakey_aux.get("invalid_q_targets")):
                    invalid_target_tensors.append(dynakey_aux["invalid_q_targets"].float().detach().flatten())

            if occupancy_tensors:
                value = torch.cat(occupancy_tensors).mean().item()
                self.log.log_scalar("dynakey/occupancy_ratio", value, it)
                metrics["dynakey/occupancy_ratio"] = value
            if active_count_tensors:
                value = torch.cat(active_count_tensors).mean().item()
                self.log.log_scalar("dynakey/active_key_count", value, it)
                metrics["dynakey/active_key_count"] = value
            if entropy_tensors:
                value = torch.cat(entropy_tensors).mean().item()
                self.log.log_scalar("dynakey/retrieval_entropy", value, it)
                metrics["dynakey/retrieval_entropy"] = value
            if fallback_tensors:
                value = torch.cat(fallback_tensors).mean().item()
                self.log.log_scalar("dynakey/identity_fallback", value, it)
                metrics["dynakey/identity_fallback"] = value
            if prediction_error_tensors:
                value = torch.cat(prediction_error_tensors).mean().item()
                self.log.log_scalar("dynakey/prediction_error", value, it)
                metrics["dynakey/prediction_error"] = value
            if residual_tensors:
                value = torch.cat(residual_tensors).mean().item()
                self.log.log_scalar("dynakey/residual_norm", value, it)
                metrics["dynakey/residual_norm"] = value
            if action_hist_tensors:
                hist = torch.stack(action_hist_tensors, dim=0).mean(dim=0)
                names = ["keep", "update", "spawn", "split", "delete"]
                for idx, name in enumerate(names):
                    value = hist[idx].item()
                    self.log.log_scalar(f"dynakey/action_{name}", value, it)
                    metrics[f"dynakey/action_{name}"] = value
            if action_count_tensors:
                counts = torch.stack(action_count_tensors, dim=0).sum(dim=0)
                names = ["keep", "update", "spawn", "split", "delete"]
                for idx, name in enumerate(names):
                    value = counts[idx].item()
                    self.log.log_scalar(f"dynakey/action_count_{name}", value, it)
                    metrics[f"dynakey/action_count_{name}"] = value
            if valid_q_tensors:
                value = torch.cat(valid_q_tensors).sum().item()
                self.log.log_scalar("dynakey/valid_q_samples", value, it)
                metrics["dynakey/valid_q_samples"] = value
            if invalid_target_tensors:
                value = torch.cat(invalid_target_tensors).sum().item()
                self.log.log_scalar("dynakey/invalid_q_targets", value, it)
                metrics["dynakey/invalid_q_targets"] = value
            if gate_tensors:
                value = torch.cat(gate_tensors).mean().item()
                self.log.log_scalar("unext_dynakey/gate_mean", value, it)
                metrics["unext_dynakey/gate_mean"] = value
            if residual_abs_tensors:
                value = torch.cat(residual_abs_tensors).mean().item()
                self.log.log_scalar("unext_dynakey/residual_abs_mean", value, it)
                metrics["unext_dynakey/residual_abs_mean"] = value
            if base_abs_tensors:
                value = torch.cat(base_abs_tensors).mean().item()
                self.log.log_scalar("unext_dynakey/base_logits_abs_mean", value, it)
                metrics["unext_dynakey/base_logits_abs_mean"] = value
            if memory_update_tensors:
                value = torch.cat(memory_update_tensors).mean().item()
                self.log.log_scalar("unext_dynakey/memory_update_rate", value, it)
                metrics["unext_dynakey/memory_update_rate"] = value
            if rejected_update_tensors:
                value = torch.cat(rejected_update_tensors).sum().item()
                self.log.log_scalar("unext_dynakey/rejected_update_count", value, it)
                metrics["unext_dynakey/rejected_update_count"] = value
            if mid_gate_tensors:
                value = torch.cat(mid_gate_tensors).mean().item()
                self.log.log_scalar("unext_dynakey/mid_memory_gate_mean", value, it)
                metrics["unext_dynakey/mid_memory_gate_mean"] = value
            if mid_contrib_tensors:
                value = torch.cat(mid_contrib_tensors).mean().item()
                self.log.log_scalar("unext_dynakey/mid_memory_contribution_norm", value, it)
                metrics["unext_dynakey/mid_memory_contribution_norm"] = value
            if enhanced_diff_tensors:
                value = torch.cat(enhanced_diff_tensors).mean().item()
                self.log.log_scalar("unext_dynakey/enhanced_feature_diff_norm", value, it)
                metrics["unext_dynakey/enhanced_feature_diff_norm"] = value
            if spatial_entropy_tensors:
                value = torch.cat(spatial_entropy_tensors).mean().item()
                self.log.log_scalar("unext_dynakey/spatial_memory_entropy", value, it)
                metrics["unext_dynakey/spatial_memory_entropy"] = value
            if metrics:
                self._log_metrics(metrics, step=it)
        except Exception:
            pass

    def _log_anchor_ode_stats(self, data, it: int) -> None:
        memory_keys = sorted(k for k in data.keys() if k.startswith("memory_aux_"))
        if not memory_keys:
            return

        try:
            metrics = {}
            prior_conf = []
            base_conf = []
            update_conf = []
            boundary_conf = []
            scale_conf = []
            slot_conf = []
            disagreement = []
            entropy = []
            final_base_residual = []
            guided_base_residual = []
            affine_abs = []
            slot_entropy = []
            slot_max_prob = []
            gate_values = []
            translate_abs = []
            scale_abs = []
            rotate_abs = []
            shear_abs = []
            for key in memory_keys:
                aux = data.get(key)
                anchor_aux = aux.get("anchor_ode_aux") if isinstance(aux, dict) else None
                if not isinstance(anchor_aux, dict):
                    continue
                for src, dst in (
                    ("confidence_prior", prior_conf),
                    ("confidence_base", base_conf),
                    ("confidence_update", update_conf),
                    ("confidence_boundary", boundary_conf),
                    ("confidence_scale", scale_conf),
                    ("effective_slot_confidence", slot_conf),
                    ("final_base_residual_abs_mean", final_base_residual),
                    ("guided_base_residual_abs_mean", guided_base_residual),
                    ("base_prior_disagreement", disagreement),
                    ("mask_entropy", entropy),
                ):
                    value = anchor_aux.get(src)
                    if torch.is_tensor(value):
                        dst.append(value.float().detach().flatten())
                affine = anchor_aux.get("affine")
                if torch.is_tensor(affine):
                    aff = affine.float().detach()
                    affine_abs.append(aff.abs().flatten())
                    translate_abs.append(aff[..., 0:2].abs().flatten())
                    scale_abs.append((aff[..., 2:4] - 1.0).abs().flatten())
                    rotate_abs.append(aff[..., 4:5].abs().flatten())
                    shear_abs.append(aff[..., 5:6].abs().flatten())
                weights = anchor_aux.get("slot_weights")
                if torch.is_tensor(weights):
                    w = weights.float().detach().clamp_min(1.0e-8)
                    slot_entropy.append((-(w * w.log()).sum(dim=-1) / math.log(max(w.shape[-1], 2))).flatten())
                    slot_max_prob.append(w.max(dim=-1).values.flatten())
                gates = anchor_aux.get("skip_gates")
                if torch.is_tensor(gates):
                    gate_values.append(gates.float().detach().flatten())

            metrics = {
                "confidence_prior_mean": prior_conf,
                "confidence_base_mean": base_conf,
                "confidence_update_mean": update_conf,
                "confidence_boundary_mean": boundary_conf,
                "confidence_scale_mean": scale_conf,
                "slot_confidence_mean": slot_conf,
                "base_prior_disagreement": disagreement,
                "mask_entropy": entropy,
                "final_base_residual_abs_mean": final_base_residual,
                "guided_base_residual_abs_mean": guided_base_residual,
                "affine_abs_mean": affine_abs,
                "translate_abs_mean": translate_abs,
                "scale_abs_mean": scale_abs,
                "rotate_abs_mean": rotate_abs,
                "shear_abs_mean": shear_abs,
                "slot_entropy": slot_entropy,
                "slot_max_prob": slot_max_prob,
            }
            if gate_values:
                gates = torch.cat(gate_values)
                metrics["gate_mean"] = [gates]
                metrics["gate_std"] = [gates.std(unbiased=False).reshape(1)]
            for name, tensors in metrics.items():
                if not tensors:
                    continue
                value = torch.cat(tensors).mean().item()
                self.log.log_scalar(f"anchor_ode/{name}", value, it)
                metrics[name] = value
            if metrics:
                logger = getattr(self, "mlflow_logger", None)
                if logger is not None:
                    logger.log_anchor_ode_diagnostics(metrics, step=it)
        except Exception:
            pass

    def _log_functional_anchor_stats(self, data, it: int) -> None:
        memory_keys = sorted(k for k in data.keys() if k.startswith("memory_aux_"))
        if not memory_keys:
            return
        try:
            buckets = {
                "residual_l1": [],
                "residual_l2": [],
                "shape_residual_norm": [],
                "boundary_residual_norm": [],
                "slot_entropy": [],
                "ed_slot_usage": [],
                "es_slot_usage": [],
                "slot_area_order_violation": [],
                "slot_order_loss": [],
                "slot_area_ed": [],
                "slot_area_early_systole": [],
                "slot_area_es": [],
                "slot_area_early_diastole": [],
                "slot_area_uncertain": [],
                "phase_entropy": [],
                "phase_source": [],
                "phase_reliability": [],
                "state_norm": [],
                "state_delta_norm": [],
                "ode_update_norm": [],
                "gate_mean_low": [],
                "gate_mean_mid": [],
                "gate_mean_high": [],
                "inject_gate_low": [],
                "inject_gate_mid": [],
                "inject_gate_high": [],
                "inject_gate_dec": [],
                "confidence_mean": [],
                "confidence_std": [],
                "trust_mean": [],
                "trust_std": [],
                "trust_spatial_std": [],
                "trust_disagreement_corr": [],
                "anchor_trust_ratio": [],
                "image_trust_ratio": [],
                "residual_abs_mean": [],
                "residual_abs_max": [],
                "residual_clip_hit_ratio": [],
                "residual_scale": [],
                "delta_abs_mean": [],
                "base_logit_abs_mean": [],
                "anchor_logit_abs_mean": [],
                "proposal_logit_abs_mean": [],
                "final_logit_abs_mean": [],
                "base_logit_std": [],
                "anchor_logit_std": [],
                "proposal_logit_std": [],
                "final_logit_std": [],
                "base_prob_mean": [],
                "anchor_prob_mean": [],
                "proposal_prob_mean": [],
                "final_prob_mean": [],
                "anchor_temperature": [],
                "early_systole_slot_usage": [],
                "early_diastole_slot_usage": [],
                "uncertain_slot_usage": [],
                "slot_max_prob": [],
            }
            for key in memory_keys:
                aux = data.get(key)
                fa = aux.get("functional_anchor_aux") if isinstance(aux, dict) else None
                if not isinstance(fa, dict):
                    continue
                for name in buckets:
                    value = fa.get(name)
                    if torch.is_tensor(value):
                        buckets[name].append(value.float().detach().flatten())
            metrics = {}
            for name, tensors in buckets.items():
                if not tensors:
                    continue
                value = torch.cat(tensors).mean().item()
                self.log.log_scalar(f"functional_anchor/{name}", value, it)
                metrics[name] = value
            if metrics:
                logger = getattr(self, "mlflow_logger", None)
                if logger is not None:
                    logger.log_functional_anchor_diagnostics(metrics, step=it)
        except Exception:
            pass

    def _log_faf_stats(self, data, it: int) -> None:
        memory_keys = sorted(k for k in data.keys() if k.startswith("memory_aux_"))
        if not memory_keys:
            return
        try:
            buckets = {
                "effective_anchor_number": [],
                "active_anchor_entropy": [],
                "top1_anchor_weight": [],
                "top3_anchor_weight_sum": [],
                "coverage_score": [],
                "coverage_gap": [],
                "anchor_function_diversity": [],
                "anchor_area_diversity": [],
                "anchor_pairwise_similarity": [],
                "write_strength_mean": [],
                "memory_update_norm": [],
                "affine_delta_norm": [],
                "affine_velocity_norm": [],
                "ode_velocity_norm": [],
                "dead_anchor_ratio": [],
                "recycled_anchor_ratio": [],
                "trust_mean": [],
                "anchor_trust_ratio": [],
                "residual_l1": [],
                "residual_l2": [],
                "residual_clip_hit_ratio": [],
                "residual_scale": [],
                "retrieval_temperature": [],
                "ode_dt": [],
            }
            for key in memory_keys:
                aux = data.get(key)
                faf = aux.get("faf_aux") if isinstance(aux, dict) else None
                if not isinstance(faf, dict):
                    continue
                for name in buckets:
                    value = faf.get(name)
                    if torch.is_tensor(value):
                        buckets[name].append(value.float().detach().flatten())
            metrics = {}
            for name, tensors in buckets.items():
                if not tensors:
                    continue
                value = torch.cat(tensors).mean().item()
                self.log.log_scalar(f"faf/{name}", value, it)
                metrics[name] = value
            if metrics:
                logger = getattr(self, "mlflow_logger", None)
                if logger is not None:
                    logger.log_faf_diagnostics(metrics, step=it)
        except Exception:
            pass

    def evaluate(
        self, val_loader, epoch, run_path, it, local_rank=None, world_size=None, full_eval: bool = False
    ):
        result = self.evaluator.evaluate(val_loader, "val", epoch, run_path, it, full_eval=full_eval)
        logger = getattr(self, "mlflow_logger", None)
        if self.main_process and logger is not None:
            logger.log_evaluation_result(result, step=it, log_artifacts=full_eval)
        return result.summary_metrics

    def test(self, test_loader, epoch, run_path, it, local_rank=None, world_size=None, full_eval: bool = False):
        result = self.evaluator.evaluate(test_loader, "test", epoch, run_path, it, full_eval=full_eval)
        logger = getattr(self, "mlflow_logger", None)
        if self.main_process and logger is not None:
            logger.log_evaluation_result(result, step=it, log_artifacts=full_eval)
        return result.summary_metrics

    def _reset_metrics(self):
        self.conf_metric.reset()

    def _threshold_candidates(self, mode: str | None = None, it: int | None = None, *, full_eval: bool = False) -> list[float]:
        eval_cfg = self.cfg.get("evaluation", {})
        mode_norm = str(mode or "").lower()
        default_threshold = float(eval_cfg.get("default_threshold", 0.5))
        active_threshold = (
            float(self.best_val_threshold)
            if mode_norm == "test" or self._best_val_threshold_ready
            else default_threshold
        )
        if not full_eval:
            if not bool(eval_cfg.get("threshold_search_during_training", True)):
                return [round(active_threshold, 2)]
            interval = int(eval_cfg.get("threshold_search_interval", 0) or 0)
            if interval > 0 and (it is None or int(it) % interval != 0):
                return [round(active_threshold, 2)]
        start = float(eval_cfg.get("threshold_search_start", 0.30))
        end = float(eval_cfg.get("threshold_search_end", 0.70))
        step = float(eval_cfg.get("threshold_search_step", 0.05))
        if step <= 0:
            return [round(active_threshold, 2)]
        values = []
        current = start
        while current <= end + 1.0e-8:
            values.append(round(current, 2))
            current += step
        return values or [round(active_threshold, 2)]

    @staticmethod
    def _threshold_key(threshold: float) -> str:
        return f"thr_{threshold:.2f}".replace(".", "p")

    @staticmethod
    def _metric_totals_template() -> dict[str, float]:
        return {
            "dice_frame_sum": 0.0,
            "dice_frame_count": 0.0,
            "iou_frame_sum": 0.0,
            "iou_frame_count": 0.0,
            "dice_video_sum": 0.0,
            "dice_video_count": 0.0,
            "iou_video_sum": 0.0,
            "iou_video_count": 0.0,
            "hd95_resized_sum": 0.0,
            "hd95_resized_count": 0.0,
            "hd95_original_sum": 0.0,
            "hd95_original_count": 0.0,
            "ed_dice_sum": 0.0,
            "ed_dice_count": 0.0,
            "es_dice_sum": 0.0,
            "es_dice_count": 0.0,
            "ed_hd95_resized_sum": 0.0,
            "ed_hd95_original_sum": 0.0,
            "ed_hd95_count": 0.0,
            "es_hd95_resized_sum": 0.0,
            "es_hd95_original_sum": 0.0,
            "es_hd95_count": 0.0,
            "assd_resized_sum": 0.0,
            "assd_resized_count": 0.0,
            "assd_original_sum": 0.0,
            "assd_original_count": 0.0,
            "precision_sum": 0.0,
            "recall_sum": 0.0,
            "acc_sum": 0.0,
            "sp_sum": 0.0,
            "F1_sum": 0.0,
            "conf_count": 0.0,
            "temporal_drift_sum": 0.0,
            "temporal_drift_count": 0.0,
            "temporal_dice_consistency_sum": 0.0,
            "temporal_dice_consistency_count": 0.0,
            "area_smoothness_sum": 0.0,
            "area_smoothness_count": 0.0,
            "centroid_jitter_sum": 0.0,
            "centroid_jitter_count": 0.0,
            "gate_mean_sum": 0.0,
            "residual_abs_mean_sum": 0.0,
            "memory_update_rate_sum": 0.0,
            "teacher_forcing_prob_sum": 0.0,
            "base_only_dice_sum": 0.0,
            "base_only_dice_count": 0.0,
            "guided_only_dice_sum": 0.0,
            "guided_only_dice_count": 0.0,
            "prior_only_dice_sum": 0.0,
            "prior_only_dice_count": 0.0,
            "functional_anchor_base_dice_sum": 0.0,
            "functional_anchor_base_dice_count": 0.0,
            "functional_anchor_anchor_dice_sum": 0.0,
            "functional_anchor_anchor_dice_count": 0.0,
            "functional_anchor_proposal_dice_sum": 0.0,
            "functional_anchor_proposal_dice_count": 0.0,
            "functional_anchor_base_area_range_sum": 0.0,
            "functional_anchor_base_area_std_sum": 0.0,
            "functional_anchor_base_area_count": 0.0,
            "functional_anchor_anchor_area_range_sum": 0.0,
            "functional_anchor_anchor_area_std_sum": 0.0,
            "functional_anchor_anchor_area_count": 0.0,
            "functional_anchor_proposal_area_range_sum": 0.0,
            "functional_anchor_proposal_area_std_sum": 0.0,
            "functional_anchor_proposal_area_count": 0.0,
            "functional_anchor_final_area_range_sum": 0.0,
            "functional_anchor_final_area_std_sum": 0.0,
            "functional_anchor_final_area_count": 0.0,
            "functional_anchor_ed_es_area_gap_sum": 0.0,
            "functional_anchor_ed_es_area_ratio_sum": 0.0,
            "functional_anchor_ed_es_area_count": 0.0,
            "functional_anchor_residual_l1_sum": 0.0,
            "functional_anchor_residual_l2_sum": 0.0,
            "functional_anchor_residual_boundary_ratio_sum": 0.0,
            "functional_anchor_shape_residual_norm_sum": 0.0,
            "functional_anchor_boundary_residual_norm_sum": 0.0,
            "functional_anchor_slot_entropy_sum": 0.0,
            "functional_anchor_ed_slot_usage_sum": 0.0,
            "functional_anchor_early_systole_slot_usage_sum": 0.0,
            "functional_anchor_es_slot_usage_sum": 0.0,
            "functional_anchor_early_diastole_slot_usage_sum": 0.0,
            "functional_anchor_uncertain_slot_usage_sum": 0.0,
            "functional_anchor_slot_max_prob_sum": 0.0,
            "functional_anchor_slot_area_order_violation_sum": 0.0,
            "functional_anchor_gate_low_sum": 0.0,
            "functional_anchor_gate_mid_sum": 0.0,
            "functional_anchor_gate_high_sum": 0.0,
            "functional_anchor_confidence_mean_sum": 0.0,
            "functional_anchor_confidence_std_sum": 0.0,
            "functional_anchor_anchor_trust_ratio_sum": 0.0,
            "functional_anchor_image_trust_ratio_sum": 0.0,
            "functional_anchor_trust_mean_sum": 0.0,
            "functional_anchor_trust_std_sum": 0.0,
            "functional_anchor_trust_spatial_std_sum": 0.0,
            "functional_anchor_trust_disagreement_corr_sum": 0.0,
            "functional_anchor_residual_abs_mean_sum": 0.0,
            "functional_anchor_residual_abs_max_sum": 0.0,
            "functional_anchor_residual_clip_hit_ratio_sum": 0.0,
            "functional_anchor_residual_scale_sum": 0.0,
            "functional_anchor_delta_abs_mean_sum": 0.0,
            "functional_anchor_base_logit_abs_mean_sum": 0.0,
            "functional_anchor_anchor_logit_abs_mean_sum": 0.0,
            "functional_anchor_proposal_logit_abs_mean_sum": 0.0,
            "functional_anchor_final_logit_abs_mean_sum": 0.0,
            "functional_anchor_base_logit_std_sum": 0.0,
            "functional_anchor_anchor_logit_std_sum": 0.0,
            "functional_anchor_proposal_logit_std_sum": 0.0,
            "functional_anchor_final_logit_std_sum": 0.0,
            "functional_anchor_base_prob_mean_sum": 0.0,
            "functional_anchor_anchor_prob_mean_sum": 0.0,
            "functional_anchor_proposal_prob_mean_sum": 0.0,
            "functional_anchor_final_prob_mean_sum": 0.0,
            "functional_anchor_anchor_temperature_sum": 0.0,
            "functional_anchor_slot_order_loss_sum": 0.0,
            "functional_anchor_slot_area_ed_sum": 0.0,
            "functional_anchor_slot_area_early_systole_sum": 0.0,
            "functional_anchor_slot_area_es_sum": 0.0,
            "functional_anchor_slot_area_early_diastole_sum": 0.0,
            "functional_anchor_slot_area_uncertain_sum": 0.0,
            "functional_anchor_phase_source_sum": 0.0,
            "functional_anchor_phase_source_metadata_ratio_sum": 0.0,
            "functional_anchor_phase_source_area_ratio_sum": 0.0,
            "functional_anchor_phase_source_time_ratio_sum": 0.0,
            "functional_anchor_phase_reliability_sum": 0.0,
            "functional_anchor_phase_reliability_std_sum": 0.0,
            "functional_anchor_phase_reliability_min_sum": 0.0,
            "functional_anchor_phase_reliability_low_ratio_sum": 0.0,
            "functional_anchor_state_norm_sum": 0.0,
            "functional_anchor_state_delta_norm_sum": 0.0,
            "functional_anchor_state_update_norm_sum": 0.0,
            "functional_anchor_state_delta_ratio_sum": 0.0,
            "functional_anchor_ode_raw_delta_norm_sum": 0.0,
            "functional_anchor_ode_clamp_ratio_sum": 0.0,
            "functional_anchor_ode_update_norm_sum": 0.0,
            "functional_anchor_inject_gate_low_sum": 0.0,
            "functional_anchor_inject_gate_mid_sum": 0.0,
            "functional_anchor_inject_gate_high_sum": 0.0,
            "functional_anchor_inject_gate_dec_sum": 0.0,
            "functional_anchor_aux_count": 0.0,
            "aux_count": 0.0,
        }

    @staticmethod
    def _batch_scalar_index(batch_data: dict, key: str, bi: int) -> int | None:
        value = batch_data.get(key)
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            item = value.detach().flatten()[bi if value.numel() > bi else 0]
            return int(item.item())
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            item = value[bi] if len(value) > bi else value[0]
            return int(item)
        try:
            return int(value)
        except Exception:
            return None

    def _resolve_phase_eval_frames(self, batch_data: dict, bi: int, sample_eval_indices: list[int]) -> tuple[int | None, int | None]:
        if not sample_eval_indices:
            return None, None
        eval_set = set(int(idx) for idx in sample_eval_indices)
        ed_idx = self._batch_scalar_index(batch_data, "ed_frame", bi)
        es_idx = self._batch_scalar_index(batch_data, "es_frame", bi)
        if ed_idx is None:
            ed_idx = sample_eval_indices[0]
        elif ed_idx not in eval_set:
            ed_idx = None
        if es_idx is None:
            es_idx = sample_eval_indices[-1]
        elif es_idx not in eval_set:
            es_idx = None
        return ed_idx, es_idx

    def _tta_cfg(self):
        return self.cfg.get("evaluation", {}).get("tta", {})

    def _tta_enabled(self) -> bool:
        tta_cfg = self._tta_cfg()
        return bool(tta_cfg.get("enabled", False)) if hasattr(tta_cfg, "get") else bool(tta_cfg)

    def _tta_modes(self) -> list[str]:
        if not self._tta_enabled():
            return ["identity"]
        modes = list(self._tta_cfg().get("modes", ["identity", "hflip"]))
        if "identity" not in modes:
            modes = ["identity"] + modes
        seen = set()
        unique_modes = []
        for mode in modes:
            mode = str(mode)
            if mode not in seen:
                seen.add(mode)
                unique_modes.append(mode)
        return unique_modes or ["identity"]

    def _clone_batch_for_forward(self, batch: dict) -> dict:
        cloned = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    @staticmethod
    def _resize_video_tensor(tensor: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
        if tensor.ndim != 5:
            return tensor
        b, t, c, _, _ = tensor.shape
        flat = tensor.reshape(b * t, c, tensor.shape[-2], tensor.shape[-1])
        kwargs = {"size": size, "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        flat = F.interpolate(flat.float(), **kwargs)
        return flat.reshape(b, t, c, size[0], size[1]).to(dtype=tensor.dtype)

    def _apply_tta_to_batch(self, batch: dict, mode: str) -> tuple[dict, tuple[int, int]]:
        aug_batch = self._clone_batch_for_forward(batch)
        original_hw = tuple(int(v) for v in aug_batch["rgb"].shape[-2:])
        if mode == "identity":
            return aug_batch, original_hw
        if mode == "hflip":
            for key in ("rgb", "ff_gt", "cls_gt"):
                if key in aug_batch and isinstance(aug_batch[key], torch.Tensor) and aug_batch[key].ndim >= 4:
                    aug_batch[key] = torch.flip(aug_batch[key], dims=[-1])
            return aug_batch, original_hw
        if mode.startswith("scale_"):
            scale = float(mode.split("_", 1)[1])
            new_h = max(1, int(round(original_hw[0] * scale)))
            new_w = max(1, int(round(original_hw[1] * scale)))
            aug_batch["rgb"] = self._resize_video_tensor(aug_batch["rgb"], (new_h, new_w), mode="bilinear")
            if "ff_gt" in aug_batch and isinstance(aug_batch["ff_gt"], torch.Tensor):
                aug_batch["ff_gt"] = self._resize_video_tensor(aug_batch["ff_gt"], (new_h, new_w), mode="nearest")
            if "cls_gt" in aug_batch and isinstance(aug_batch["cls_gt"], torch.Tensor):
                aug_batch["cls_gt"] = self._resize_video_tensor(aug_batch["cls_gt"], (new_h, new_w), mode="nearest")
            return aug_batch, original_hw
        raise ValueError(f"Unsupported TTA mode: {mode}")

    def _invert_tta_tensor(self, tensor: torch.Tensor, mode: str, original_hw: tuple[int, int]) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            return tensor
        out = tensor
        if mode == "hflip":
            out = torch.flip(out, dims=[-1])
        elif mode.startswith("scale_"):
            out = F.interpolate(out.float(), size=original_hw, mode="bilinear", align_corners=False).to(dtype=tensor.dtype)
        return out

    def _forward_eval_with_tta(self, batch: dict) -> dict:
        modes = self._tta_modes()
        if modes == ["identity"]:
            out = self.model(batch)
            return self._ensure_finite_outputs(out)

        identity_out = None
        logits_sums = {}
        logits_counts = {}
        mask_sums = {}
        mask_counts = {}
        for mode in modes:
            aug_batch, original_hw = self._apply_tta_to_batch(batch, mode)
            out = self.model(aug_batch)
            out = self._ensure_finite_outputs(out)
            if identity_out is None:
                identity_out = out
            for key, value in out.items():
                if not torch.is_tensor(value):
                    continue
                if key.startswith("logits_"):
                    restored = self._invert_tta_tensor(value, mode, original_hw)
                    if key not in logits_sums:
                        logits_sums[key] = restored.float()
                        logits_counts[key] = 1
                    else:
                        logits_sums[key] = logits_sums[key] + restored.float()
                        logits_counts[key] += 1
                    continue
                if key.startswith("masks_"):
                    restored = self._invert_tta_tensor(value, mode, original_hw)
                    if key not in mask_sums:
                        mask_sums[key] = restored.float()
                        mask_counts[key] = 1
                    else:
                        mask_sums[key] = mask_sums[key] + restored.float()
                        mask_counts[key] += 1

        combined = dict(identity_out or {})
        for key, value in logits_sums.items():
            avg_logits = value / max(logits_counts.get(key, 1), 1)
            ref = combined.get(key)
            combined[key] = avg_logits.to(dtype=ref.dtype if torch.is_tensor(ref) else avg_logits.dtype)
            frame_id = key.split("_", 1)[1]
            mask_key = f"masks_{frame_id}"
            if avg_logits.shape[1] > 1:
                avg_masks = torch.softmax(avg_logits, dim=1)[:, 1:]
            else:
                avg_masks = torch.sigmoid(avg_logits)
            mask_ref = combined.get(mask_key)
            combined[mask_key] = avg_masks.to(dtype=mask_ref.dtype if torch.is_tensor(mask_ref) else avg_masks.dtype)

        for key, value in mask_sums.items():
            if key in combined and key.replace("masks_", "logits_") in logits_sums:
                continue
            avg = value / max(mask_counts.get(key, 1), 1)
            ref = combined.get(key)
            combined[key] = avg.to(dtype=ref.dtype if torch.is_tensor(ref) else avg.dtype)
        return combined

    def _swap_to_ema_for_eval(self):
        if self.ema is None or not self.ema_eval:
            return None
        raw_state = {
            key: value.detach().clone()
            for key, value in self.model_without_ddp.state_dict().items()
        }
        self.model_without_ddp.load_state_dict(self.ema.state_dict(), strict=True)
        return raw_state

    def _restore_model_state(self, raw_state) -> None:
        if raw_state is not None:
            self.model_without_ddp.load_state_dict(raw_state, strict=True)

    def _save_best_if_needed(self, mode: str, metrics: dict, epoch: int, it: int, raw_state=None) -> None:
        if not self.main_process or mode not in {"val", "validation"}:
            return
        metric_name = str(self.cfg.get("evaluation", {}).get("best_ckpt_metric", "best_threshold_dice_frame_mean"))
        metric = float(metrics.get(metric_name, metrics.get("dice_frame_mean", 0.0)))
        if metric <= self.best_val_metric:
            return
        self.best_val_metric = metric
        self.run_path.mkdir(parents=True, exist_ok=True)
        current_state = self.model_without_ddp.state_dict()
        raw_to_save = raw_state if raw_state is not None else current_state
        if self.ema is not None and self.ema_eval:
            torch.save(raw_to_save, self.run_path / "best_raw.pth")
            torch.save(self.ema.state_dict(), self.run_path / "best_ema.pth")
            saved_weight_files = ["best_raw.pth", "best_ema.pth"]
        else:
            torch.save(raw_to_save, self.run_path / "best_raw.pth")
            saved_weight_files = ["best_raw.pth"]
        metadata = {
            "iteration": int(it),
            "epoch": int(epoch),
            "metric_name": metric_name,
            "metric": metric,
            "best_val_threshold": float(metrics.get("best_val_threshold", self.best_val_threshold)),
            "ema_enabled": self.ema_enabled,
            "ema_eval": self.ema_eval,
            "tta_enabled": self._tta_enabled(),
            "tta_modes": self._tta_modes(),
            "postprocess_enabled": self._postprocess_enabled(),
            "saved_weight_files": saved_weight_files,
        }
        with (self.run_path / "best_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        logger = getattr(self, "mlflow_logger", None)
        if logger is not None:
            logger.log_best(metrics, epoch=epoch, iteration=it)
            for filename in saved_weight_files:
                logger.log_checkpoint(self.run_path / filename, artifact_name=filename)
            logger.log_checkpoint(self.run_path / "best_summary.json", artifact_name="best_summary.json")
        self.log.info(f"Saved best checkpoint set at iter {it}: {metric_name}={metric:.6f}")

    def _run_evaluation_impl(self, data_loader, mode, epoch, run_path, it, *, full_eval: bool = False) -> EvaluationResult:
        if self.is_distributed:
            dist.barrier()

        if self.main_process:
            self.log.info(
                f"[{mode.capitalize()}] Iter {it} Epoch {epoch}: Start Evaluation..."
            )

        prev_mode = self.model.training
        self.model.eval()
        raw_state_for_restore = self._swap_to_ema_for_eval()
        self._reset_metrics()
        threshold_candidates = self._threshold_candidates(mode, it, full_eval=full_eval)
        active_threshold = 0.5
        if mode == "test":
            if not self._best_val_threshold_ready and self.main_process:
                self.log.warning("[Test] best_val_threshold is not available; using 0.5.")
            active_threshold = float(self.best_val_threshold)
        elif mode not in {"val", "validation"}:
            active_threshold = float(self.best_val_threshold if self._best_val_threshold_ready else 0.5)

        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        visual_artifacts = []
        per_frame_metrics = []
        per_video_metrics = []
        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(data_loader):
                    self._move_to_device(batch_data)
                    batch_data["init_mode"] = self._resolve_phase_init(mode)

                    with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                        out = self._forward_eval_with_tta(batch_data)

                    supervised_indices = self._resolve_supervised_indices(batch_data)
                    eval_indices = self._resolve_eval_indices(batch_data)
                    required_eval_ids = sorted(torch.nonzero(eval_indices.any(dim=0), as_tuple=False).flatten().tolist())
                    mask_keys = [f"masks_{ti}" for ti in required_eval_ids]

                    if batch_idx == 0 and self.main_process:
                        valid_mask_counts = eval_indices.sum(dim=1).detach().cpu().tolist()
                        uses_oracle_gt = batch_data["init_mode"] == "oracle_gt" and self._oracle_init_allowed()
                        self.log.info(
                            f"[{mode.capitalize()}] init_mode={batch_data['init_mode']} | "
                            f"oracle_gt_init_allowed={self._oracle_init_allowed()} | "
                            f"uses_oracle_gt={uses_oracle_gt} | "
                            f"metric_space={str(self.cfg.get('evaluation', {}).get('metric_space', 'original'))} | "
                            f"exclude_init_frame={bool(self.cfg.get('evaluation', {}).get('exclude_init_frame', False))} | "
                            f"supervised_indices={self._format_frame_mask(supervised_indices)} | "
                            f"eval_indices={self._format_frame_mask(eval_indices)} | "
                            f"valid_mask_frame_counts={valid_mask_counts}"
                        )

                    if not all(k in out for k in mask_keys):
                        continue

                    gt = batch_data["cls_gt"]
                    if gt.dim() == 5:
                        gt = gt.squeeze(2)

                    batch_size = gt.shape[0]
                    if batch_idx == 0:
                        metric_totals = {
                            "dice_frame_sum": 0.0,
                            "dice_frame_count": 0.0,
                            "iou_frame_sum": 0.0,
                            "iou_frame_count": 0.0,
                            "dice_video_sum": 0.0,
                            "dice_video_count": 0.0,
                            "iou_video_sum": 0.0,
                            "iou_video_count": 0.0,
                            "hd95_resized_sum": 0.0,
                            "hd95_resized_count": 0.0,
                            "hd95_original_sum": 0.0,
                            "hd95_original_count": 0.0,
                            "assd_resized_sum": 0.0,
                            "assd_resized_count": 0.0,
                            "assd_original_sum": 0.0,
                            "assd_original_count": 0.0,
                            "precision_sum": 0.0,
                            "recall_sum": 0.0,
                            "acc_sum": 0.0,
                            "sp_sum": 0.0,
                            "F1_sum": 0.0,
                            "conf_count": 0.0,
                            "temporal_drift_sum": 0.0,
                            "temporal_drift_count": 0.0,
                            "temporal_dice_consistency_sum": 0.0,
                            "temporal_dice_consistency_count": 0.0,
                            "area_smoothness_sum": 0.0,
                            "area_smoothness_count": 0.0,
                            "centroid_jitter_sum": 0.0,
                            "centroid_jitter_count": 0.0,
                            "gate_mean_sum": 0.0,
                            "residual_abs_mean_sum": 0.0,
                            "memory_update_rate_sum": 0.0,
                            "teacher_forcing_prob_sum": 0.0,
                            "base_only_dice_sum": 0.0,
                            "base_only_dice_count": 0.0,
                            "guided_only_dice_sum": 0.0,
                            "guided_only_dice_count": 0.0,
                            "prior_only_dice_sum": 0.0,
                            "prior_only_dice_count": 0.0,
                            "functional_anchor_base_dice_sum": 0.0,
                            "functional_anchor_base_dice_count": 0.0,
                            "functional_anchor_anchor_dice_sum": 0.0,
                            "functional_anchor_anchor_dice_count": 0.0,
                            "functional_anchor_proposal_dice_sum": 0.0,
                            "functional_anchor_proposal_dice_count": 0.0,
                            "functional_anchor_base_area_range_sum": 0.0,
                            "functional_anchor_base_area_std_sum": 0.0,
                            "functional_anchor_base_area_count": 0.0,
                            "functional_anchor_anchor_area_range_sum": 0.0,
                            "functional_anchor_anchor_area_std_sum": 0.0,
                            "functional_anchor_anchor_area_count": 0.0,
                            "functional_anchor_proposal_area_range_sum": 0.0,
                            "functional_anchor_proposal_area_std_sum": 0.0,
                            "functional_anchor_proposal_area_count": 0.0,
                            "functional_anchor_final_area_range_sum": 0.0,
                            "functional_anchor_final_area_std_sum": 0.0,
                            "functional_anchor_final_area_count": 0.0,
                            "functional_anchor_ed_es_area_gap_sum": 0.0,
                            "functional_anchor_ed_es_area_ratio_sum": 0.0,
                            "functional_anchor_ed_es_area_count": 0.0,
                            "functional_anchor_residual_l1_sum": 0.0,
                            "functional_anchor_residual_l2_sum": 0.0,
                            "functional_anchor_residual_boundary_ratio_sum": 0.0,
                            "functional_anchor_shape_residual_norm_sum": 0.0,
                            "functional_anchor_boundary_residual_norm_sum": 0.0,
                            "functional_anchor_slot_entropy_sum": 0.0,
                            "functional_anchor_ed_slot_usage_sum": 0.0,
                            "functional_anchor_early_systole_slot_usage_sum": 0.0,
                            "functional_anchor_es_slot_usage_sum": 0.0,
                            "functional_anchor_early_diastole_slot_usage_sum": 0.0,
                            "functional_anchor_uncertain_slot_usage_sum": 0.0,
                            "functional_anchor_slot_max_prob_sum": 0.0,
                            "functional_anchor_slot_area_order_violation_sum": 0.0,
                            "functional_anchor_gate_low_sum": 0.0,
                            "functional_anchor_gate_mid_sum": 0.0,
                            "functional_anchor_gate_high_sum": 0.0,
                            "functional_anchor_confidence_mean_sum": 0.0,
                            "functional_anchor_confidence_std_sum": 0.0,
                            "functional_anchor_anchor_trust_ratio_sum": 0.0,
                            "functional_anchor_image_trust_ratio_sum": 0.0,
                            "functional_anchor_trust_mean_sum": 0.0,
                            "functional_anchor_trust_std_sum": 0.0,
                            "functional_anchor_trust_spatial_std_sum": 0.0,
                            "functional_anchor_trust_disagreement_corr_sum": 0.0,
                            "functional_anchor_residual_abs_mean_sum": 0.0,
                            "functional_anchor_residual_abs_max_sum": 0.0,
                            "functional_anchor_residual_clip_hit_ratio_sum": 0.0,
                            "functional_anchor_residual_scale_sum": 0.0,
                            "functional_anchor_delta_abs_mean_sum": 0.0,
                            "functional_anchor_base_logit_abs_mean_sum": 0.0,
                            "functional_anchor_anchor_logit_abs_mean_sum": 0.0,
                            "functional_anchor_proposal_logit_abs_mean_sum": 0.0,
                            "functional_anchor_final_logit_abs_mean_sum": 0.0,
                            "functional_anchor_base_logit_std_sum": 0.0,
                            "functional_anchor_anchor_logit_std_sum": 0.0,
                            "functional_anchor_proposal_logit_std_sum": 0.0,
                            "functional_anchor_final_logit_std_sum": 0.0,
                            "functional_anchor_base_prob_mean_sum": 0.0,
                            "functional_anchor_anchor_prob_mean_sum": 0.0,
                            "functional_anchor_proposal_prob_mean_sum": 0.0,
                            "functional_anchor_final_prob_mean_sum": 0.0,
                            "functional_anchor_anchor_temperature_sum": 0.0,
                            "functional_anchor_slot_order_loss_sum": 0.0,
                            "functional_anchor_slot_area_ed_sum": 0.0,
                            "functional_anchor_slot_area_early_systole_sum": 0.0,
                            "functional_anchor_slot_area_es_sum": 0.0,
                            "functional_anchor_slot_area_early_diastole_sum": 0.0,
                            "functional_anchor_slot_area_uncertain_sum": 0.0,
                            "functional_anchor_phase_source_sum": 0.0,
                            "functional_anchor_phase_source_metadata_ratio_sum": 0.0,
                            "functional_anchor_phase_source_area_ratio_sum": 0.0,
                            "functional_anchor_phase_source_time_ratio_sum": 0.0,
                            "functional_anchor_phase_reliability_sum": 0.0,
                            "functional_anchor_phase_reliability_std_sum": 0.0,
                            "functional_anchor_phase_reliability_min_sum": 0.0,
                            "functional_anchor_phase_reliability_low_ratio_sum": 0.0,
                            "functional_anchor_state_norm_sum": 0.0,
                            "functional_anchor_state_delta_norm_sum": 0.0,
                            "functional_anchor_state_update_norm_sum": 0.0,
                            "functional_anchor_state_delta_ratio_sum": 0.0,
                            "functional_anchor_ode_raw_delta_norm_sum": 0.0,
                            "functional_anchor_ode_clamp_ratio_sum": 0.0,
                            "functional_anchor_ode_update_norm_sum": 0.0,
                            "functional_anchor_inject_gate_low_sum": 0.0,
                            "functional_anchor_inject_gate_mid_sum": 0.0,
                            "functional_anchor_inject_gate_high_sum": 0.0,
                            "functional_anchor_inject_gate_dec_sum": 0.0,
                            "functional_anchor_aux_count": 0.0,
                            "aux_count": 0.0,
                        }
                        for thr in threshold_candidates:
                            key = self._threshold_key(thr)
                            metric_totals[f"{key}_dice_sum"] = 0.0
                            metric_totals[f"{key}_dice_count"] = 0.0
                        for key, value in self._metric_totals_template().items():
                            metric_totals.setdefault(key, value)

                    conf_pred_frames = []
                    conf_gt_frames = []

                    for bi in range(batch_size):
                        sample_dice = []
                        sample_iou = []
                        drift_values = []
                        base_area_values = []
                        anchor_area_values = []
                        proposal_area_values = []
                        final_area_values = []
                        original_sizes = batch_data.get("original_size")
                        sample_eval_indices = torch.nonzero(eval_indices[bi], as_tuple=False).flatten().tolist()
                        ed_frame, es_frame = self._resolve_phase_eval_frames(batch_data, bi, sample_eval_indices)
                        for ti in sample_eval_indices:
                            pred = out[f"masks_{ti}"][bi:bi + 1]
                            if pred.shape[1] > 1:
                                pred = pred[:, 1:2, ...]
                            for thr in threshold_candidates:
                                thr_bin = self._postprocess_binary_mask((pred > thr).float())
                                thr_dice, _ = self._binary_overlap_metrics(thr_bin, gt[bi, ti, ...].unsqueeze(0).unsqueeze(0).float())
                                key = self._threshold_key(thr)
                                metric_totals[f"{key}_dice_sum"] += thr_dice
                                metric_totals[f"{key}_dice_count"] += 1.0
                            pred_bin = self._postprocess_binary_mask((pred > active_threshold).float())
                            gt_frame = gt[bi, ti, ...].unsqueeze(0).unsqueeze(0).float()

                            memory_aux = out.get(f"memory_aux_{ti}")
                            anchor_aux = memory_aux.get("anchor_ode_aux") if isinstance(memory_aux, dict) else None
                            if isinstance(anchor_aux, dict):
                                for src, prefix in (
                                    ("base_object_logits", "base_only"),
                                    ("guided_object_logits", "guided_only"),
                                    ("prior_logits", "prior_only"),
                                ):
                                    aux_logits = anchor_aux.get(src)
                                    if torch.is_tensor(aux_logits) and aux_logits.shape[0] > bi:
                                        aux_prob = torch.sigmoid(aux_logits[bi : bi + 1]).max(dim=1, keepdim=True).values
                                        aux_bin = self._postprocess_binary_mask((aux_prob > active_threshold).float())
                                        aux_dice, _ = self._binary_overlap_metrics(aux_bin, gt_frame)
                                        metric_totals[f"{prefix}_dice_sum"] += aux_dice
                                        metric_totals[f"{prefix}_dice_count"] += 1.0
                            functional_aux = memory_aux.get("functional_anchor_aux") if isinstance(memory_aux, dict) else None
                            if isinstance(functional_aux, dict):
                                required_logits = (
                                    "base_object_logits",
                                    "anchor_logits",
                                    "proposal_logits",
                                    "final_object_logits",
                                )
                                missing_logits = [name for name in required_logits if not torch.is_tensor(functional_aux.get(name))]
                                if missing_logits:
                                    raise KeyError(f"Missing Functional-UNeXt logits in functional_anchor_aux: {missing_logits}")
                                for src, prefix in (
                                    ("base_object_logits", "functional_anchor_base"),
                                    ("anchor_logits", "functional_anchor_anchor"),
                                    ("proposal_logits", "functional_anchor_proposal"),
                                ):
                                    aux_logits = functional_aux.get(src)
                                    if torch.is_tensor(aux_logits) and aux_logits.shape[0] > bi:
                                        aux_prob = torch.sigmoid(aux_logits[bi : bi + 1, :1])
                                        aux_bin = self._postprocess_binary_mask((aux_prob > active_threshold).float())
                                        aux_dice, _ = self._binary_overlap_metrics(aux_bin, gt_frame)
                                        metric_totals[f"{prefix}_dice_sum"] += aux_dice
                                        metric_totals[f"{prefix}_dice_count"] += 1.0
                                        area_value = aux_bin.float().mean().detach()
                                        if prefix == "functional_anchor_base":
                                            base_area_values.append(area_value)
                                        elif prefix == "functional_anchor_anchor":
                                            anchor_area_values.append(area_value)
                                        elif prefix == "functional_anchor_proposal":
                                            proposal_area_values.append(area_value)
                                final_area_values.append(pred_bin.float().mean().detach())
                                residual = functional_aux.get("residual_logits")
                                boundary = functional_aux.get("boundary_residual_logits")
                                shape = functional_aux.get("shape_residual_logits")
                                if torch.is_tensor(residual):
                                    item = residual[bi : bi + 1].float()
                                    metric_totals["functional_anchor_residual_l1_sum"] += float(item.abs().mean().item())
                                    metric_totals["functional_anchor_residual_l2_sum"] += float(item.pow(2).mean().sqrt().item())
                                if torch.is_tensor(boundary):
                                    item = boundary[bi : bi + 1].float()
                                    metric_totals["functional_anchor_boundary_residual_norm_sum"] += float(item.abs().mean().item())
                                if torch.is_tensor(shape):
                                    item = shape[bi : bi + 1].float()
                                    metric_totals["functional_anchor_shape_residual_norm_sum"] += float(item.abs().mean().item())
                                if torch.is_tensor(boundary) and torch.is_tensor(residual):
                                    ratio = boundary[bi : bi + 1].float().abs().mean() / residual[bi : bi + 1].float().abs().mean().clamp_min(1.0e-6)
                                    metric_totals["functional_anchor_residual_boundary_ratio_sum"] += float(ratio.item())
                                for src, dst in (
                                    ("slot_entropy", "functional_anchor_slot_entropy_sum"),
                                    ("ed_slot_usage", "functional_anchor_ed_slot_usage_sum"),
                                    ("early_systole_slot_usage", "functional_anchor_early_systole_slot_usage_sum"),
                                    ("es_slot_usage", "functional_anchor_es_slot_usage_sum"),
                                    ("early_diastole_slot_usage", "functional_anchor_early_diastole_slot_usage_sum"),
                                    ("uncertain_slot_usage", "functional_anchor_uncertain_slot_usage_sum"),
                                    ("slot_max_prob", "functional_anchor_slot_max_prob_sum"),
                                    ("slot_area_order_violation", "functional_anchor_slot_area_order_violation_sum"),
                                    ("slot_order_loss", "functional_anchor_slot_order_loss_sum"),
                                    ("slot_area_ed", "functional_anchor_slot_area_ed_sum"),
                                    ("slot_area_early_systole", "functional_anchor_slot_area_early_systole_sum"),
                                    ("slot_area_es", "functional_anchor_slot_area_es_sum"),
                                    ("slot_area_early_diastole", "functional_anchor_slot_area_early_diastole_sum"),
                                    ("slot_area_uncertain", "functional_anchor_slot_area_uncertain_sum"),
                                    ("phase_source", "functional_anchor_phase_source_sum"),
                                    ("phase_reliability", "functional_anchor_phase_reliability_sum"),
                                    ("state_norm", "functional_anchor_state_norm_sum"),
                                    ("state_delta_norm", "functional_anchor_state_delta_norm_sum"),
                                    ("ode_update_norm", "functional_anchor_ode_update_norm_sum"),
                                    ("gate_mean_low", "functional_anchor_gate_low_sum"),
                                    ("gate_mean_mid", "functional_anchor_gate_mid_sum"),
                                    ("gate_mean_high", "functional_anchor_gate_high_sum"),
                                    ("inject_gate_low", "functional_anchor_inject_gate_low_sum"),
                                    ("inject_gate_mid", "functional_anchor_inject_gate_mid_sum"),
                                    ("inject_gate_high", "functional_anchor_inject_gate_high_sum"),
                                    ("inject_gate_dec", "functional_anchor_inject_gate_dec_sum"),
                                    ("confidence_mean", "functional_anchor_confidence_mean_sum"),
                                    ("confidence_std", "functional_anchor_confidence_std_sum"),
                                    ("trust_mean", "functional_anchor_trust_mean_sum"),
                                    ("trust_std", "functional_anchor_trust_std_sum"),
                                    ("trust_spatial_std", "functional_anchor_trust_spatial_std_sum"),
                                    ("trust_disagreement_corr", "functional_anchor_trust_disagreement_corr_sum"),
                                    ("residual_abs_mean", "functional_anchor_residual_abs_mean_sum"),
                                    ("residual_abs_max", "functional_anchor_residual_abs_max_sum"),
                                    ("residual_clip_hit_ratio", "functional_anchor_residual_clip_hit_ratio_sum"),
                                    ("residual_scale", "functional_anchor_residual_scale_sum"),
                                    ("delta_abs_mean", "functional_anchor_delta_abs_mean_sum"),
                                    ("base_logit_abs_mean", "functional_anchor_base_logit_abs_mean_sum"),
                                    ("anchor_logit_abs_mean", "functional_anchor_anchor_logit_abs_mean_sum"),
                                    ("proposal_logit_abs_mean", "functional_anchor_proposal_logit_abs_mean_sum"),
                                    ("final_logit_abs_mean", "functional_anchor_final_logit_abs_mean_sum"),
                                    ("base_logit_std", "functional_anchor_base_logit_std_sum"),
                                    ("anchor_logit_std", "functional_anchor_anchor_logit_std_sum"),
                                    ("proposal_logit_std", "functional_anchor_proposal_logit_std_sum"),
                                    ("final_logit_std", "functional_anchor_final_logit_std_sum"),
                                    ("base_prob_mean", "functional_anchor_base_prob_mean_sum"),
                                    ("anchor_prob_mean", "functional_anchor_anchor_prob_mean_sum"),
                                    ("proposal_prob_mean", "functional_anchor_proposal_prob_mean_sum"),
                                    ("final_prob_mean", "functional_anchor_final_prob_mean_sum"),
                                    ("anchor_temperature", "functional_anchor_anchor_temperature_sum"),
                                    ("anchor_trust_ratio", "functional_anchor_anchor_trust_ratio_sum"),
                                    ("image_trust_ratio", "functional_anchor_image_trust_ratio_sum"),
                                    ("phase_source_metadata_ratio", "functional_anchor_phase_source_metadata_ratio_sum"),
                                    ("phase_source_area_ratio", "functional_anchor_phase_source_area_ratio_sum"),
                                    ("phase_source_time_ratio", "functional_anchor_phase_source_time_ratio_sum"),
                                    ("state_update_norm", "functional_anchor_state_update_norm_sum"),
                                    ("state_delta_ratio", "functional_anchor_state_delta_ratio_sum"),
                                    ("ode_raw_delta_norm", "functional_anchor_ode_raw_delta_norm_sum"),
                                    ("ode_clamp_ratio", "functional_anchor_ode_clamp_ratio_sum"),
                                ):
                                    value = functional_aux.get(src)
                                    if torch.is_tensor(value):
                                        metric_totals[dst] += float(value.float().mean().item())
                                phase_rel = functional_aux.get("phase_reliability")
                                if torch.is_tensor(phase_rel):
                                    rel = phase_rel.float()
                                    metric_totals["functional_anchor_phase_reliability_std_sum"] += float(rel.std(unbiased=False).item())
                                    metric_totals["functional_anchor_phase_reliability_min_sum"] += float(rel.min().item())
                                    metric_totals["functional_anchor_phase_reliability_low_ratio_sum"] += float((rel < 0.5).float().mean().item())
                                metric_totals["functional_anchor_aux_count"] += 1.0

                            faf_aux = memory_aux.get("faf_aux") if isinstance(memory_aux, dict) else None
                            if isinstance(faf_aux, dict):
                                for src, prefix in (
                                    ("base_object_logits", "faf_base"),
                                    ("proposal_logits", "faf_anchor"),
                                ):
                                    aux_logits = faf_aux.get(src)
                                    if torch.is_tensor(aux_logits) and aux_logits.shape[0] > bi:
                                        aux_prob = torch.sigmoid(aux_logits[bi : bi + 1, :1])
                                        aux_bin = self._postprocess_binary_mask((aux_prob > active_threshold).float())
                                        aux_dice, _ = self._binary_overlap_metrics(aux_bin, gt_frame)
                                        metric_totals[f"{prefix}_dice_sum"] = metric_totals.get(f"{prefix}_dice_sum", 0.0) + aux_dice
                                        metric_totals[f"{prefix}_dice_count"] = metric_totals.get(f"{prefix}_dice_count", 0.0) + 1.0
                                anchor_proposals = faf_aux.get("anchor_proposals")
                                active_weights = faf_aux.get("active_weights")
                                if torch.is_tensor(anchor_proposals) and anchor_proposals.shape[0] > bi:
                                    proposals = anchor_proposals[bi : bi + 1, :1]
                                    proposal_dices = []
                                    for ai in range(proposals.shape[2]):
                                        prop_prob = torch.sigmoid(proposals[:, :, ai])
                                        prop_bin = self._postprocess_binary_mask((prop_prob > active_threshold).float())
                                        prop_dice, _ = self._binary_overlap_metrics(prop_bin, gt_frame)
                                        proposal_dices.append(prop_dice)
                                    if proposal_dices:
                                        dice_tensor = torch.tensor(proposal_dices, device=pred.device, dtype=torch.float32)
                                        metric_totals["faf_proposal_oracle_dice_sum"] = metric_totals.get("faf_proposal_oracle_dice_sum", 0.0) + float(dice_tensor.max().item())
                                        metric_totals["faf_proposal_mean_dice_sum"] = metric_totals.get("faf_proposal_mean_dice_sum", 0.0) + float(dice_tensor.mean().item())
                                        metric_totals["faf_proposal_dice_count"] = metric_totals.get("faf_proposal_dice_count", 0.0) + 1.0
                                        if torch.is_tensor(active_weights) and active_weights.shape[0] > bi:
                                            top_idx = int(active_weights[bi, 0].argmax().item())
                                            if top_idx < len(proposal_dices):
                                                metric_totals["faf_proposal_top1_dice_sum"] = metric_totals.get("faf_proposal_top1_dice_sum", 0.0) + proposal_dices[top_idx]
                                                metric_totals["faf_proposal_top1_dice_count"] = metric_totals.get("faf_proposal_top1_dice_count", 0.0) + 1.0
                                for src, dst in (
                                    ("effective_anchor_number", "faf_effective_anchor_number_sum"),
                                    ("active_anchor_entropy", "faf_active_anchor_entropy_sum"),
                                    ("top1_anchor_weight", "faf_top1_anchor_weight_sum"),
                                    ("top3_anchor_weight_sum", "faf_top3_anchor_weight_sum"),
                                    ("coverage_score", "faf_coverage_score_sum"),
                                    ("coverage_gap", "faf_coverage_gap_sum"),
                                    ("anchor_function_diversity", "faf_anchor_function_diversity_sum"),
                                    ("anchor_area_diversity", "faf_anchor_area_diversity_sum"),
                                    ("anchor_pairwise_similarity", "faf_anchor_pairwise_similarity_sum"),
                                    ("write_strength_mean", "faf_write_strength_mean_sum"),
                                    ("memory_update_norm", "faf_memory_update_norm_sum"),
                                    ("affine_delta_norm", "faf_affine_delta_norm_sum"),
                                    ("affine_velocity_norm", "faf_affine_velocity_norm_sum"),
                                    ("ode_velocity_norm", "faf_ode_velocity_norm_sum"),
                                    ("dead_anchor_ratio", "faf_dead_anchor_ratio_sum"),
                                    ("recycled_anchor_ratio", "faf_recycled_anchor_ratio_sum"),
                                    ("trust_mean", "faf_trust_mean_sum"),
                                    ("anchor_trust_ratio", "faf_anchor_trust_ratio_sum"),
                                    ("residual_l1", "faf_residual_l1_sum"),
                                    ("residual_l2", "faf_residual_l2_sum"),
                                    ("residual_clip_hit_ratio", "faf_residual_clip_hit_ratio_sum"),
                                    ("residual_scale", "faf_residual_scale_sum"),
                                    ("retrieval_temperature", "faf_retrieval_temperature_sum"),
                                    ("ode_dt", "faf_ode_dt_sum"),
                                ):
                                    value = faf_aux.get(src)
                                    if torch.is_tensor(value):
                                        metric_totals[dst] = metric_totals.get(dst, 0.0) + float(value.float().mean().item())
                                metric_totals["faf_aux_count"] = metric_totals.get("faf_aux_count", 0.0) + 1.0

                            dice_t, iou_t = self._binary_overlap_metrics(pred_bin, gt_frame)
                            metric_totals["dice_frame_sum"] += dice_t
                            metric_totals["dice_frame_count"] += 1.0
                            metric_totals["iou_frame_sum"] += iou_t
                            metric_totals["iou_frame_count"] += 1.0
                            sample_dice.append(dice_t)
                            sample_iou.append(iou_t)

                            hd95_resized, assd_resized = self._surface_metrics_single(pred_bin, gt_frame)
                            metric_totals["hd95_resized_sum"] += hd95_resized
                            metric_totals["hd95_resized_count"] += 1.0
                            metric_totals["assd_resized_sum"] += assd_resized
                            metric_totals["assd_resized_count"] += 1.0

                            original_hw = [pred.shape[-2], pred.shape[-1]]
                            if original_sizes is not None:
                                if original_sizes.dim() == 3:
                                    original_hw = original_sizes[bi, ti].tolist()
                                else:
                                    original_hw = original_sizes[bi].tolist()
                            pred_orig, gt_orig = self._resize_to_original(pred_bin, gt_frame, original_hw)
                            hd95_original, assd_original = self._surface_metrics_single(pred_orig, gt_orig)
                            metric_totals["hd95_original_sum"] += hd95_original
                            metric_totals["hd95_original_count"] += 1.0
                            metric_totals["assd_original_sum"] += assd_original
                            metric_totals["assd_original_count"] += 1.0

                            phase_names = []
                            if ed_frame is not None and int(ti) == int(ed_frame):
                                phase_names.append("ed")
                            if es_frame is not None and int(ti) == int(es_frame):
                                phase_names.append("es")
                            for phase_name in phase_names:
                                metric_totals[f"{phase_name}_dice_sum"] += dice_t
                                metric_totals[f"{phase_name}_dice_count"] += 1.0
                                metric_totals[f"{phase_name}_hd95_resized_sum"] += hd95_resized
                                metric_totals[f"{phase_name}_hd95_original_sum"] += hd95_original
                                metric_totals[f"{phase_name}_hd95_count"] += 1.0

                            if self.main_process:
                                sample_name = ""
                                info = batch_data.get("info", {})
                                if isinstance(info, dict) and "name" in info:
                                    name_value = info["name"]
                                    sample_name = str(name_value[bi] if isinstance(name_value, (list, tuple)) and len(name_value) > bi else name_value)
                                per_frame_metrics.append(
                                    {
                                        "mode": mode,
                                        "iteration": int(it),
                                        "video": sample_name,
                                        "sample_index": int(bi),
                                        "frame": int(ti),
                                        "phase": ",".join(phase_names),
                                        "dice": float(dice_t),
                                        "iou": float(iou_t),
                                        "hd95_resized": float(hd95_resized),
                                        "hd95_original": float(hd95_original),
                                    }
                                )

                            conf_pred_frames.append(pred_bin)
                            conf_gt_frames.append(gt_frame)

                        all_mask_keys = sorted(
                            (key for key in out.keys() if key.startswith("masks_")),
                            key=lambda key: int(key.split("_")[-1]),
                        )
                        prev_pred = None
                        area_values = []
                        centroid_values = []
                        for key in all_mask_keys:
                            pred_any = out[key][bi:bi + 1]
                            if pred_any.shape[1] > 1:
                                pred_any = pred_any[:, 1:2, ...]
                            pred_any = self._postprocess_binary_mask((pred_any > active_threshold).float())
                            area_values.append(pred_any.mean().detach())
                            mass = pred_any.sum(dim=(-2, -1)).clamp_min(1.0)
                            yy = torch.linspace(0.0, 1.0, pred_any.shape[-2], device=pred_any.device).view(1, 1, -1, 1)
                            xx = torch.linspace(0.0, 1.0, pred_any.shape[-1], device=pred_any.device).view(1, 1, 1, -1)
                            cy = (pred_any * yy).sum(dim=(-2, -1)) / mass
                            cx = (pred_any * xx).sum(dim=(-2, -1)) / mass
                            centroid_values.append(torch.stack([cx.flatten()[0], cy.flatten()[0]]))
                            if prev_pred is not None:
                                _, iou_prev = self._binary_overlap_metrics(pred_any, prev_pred)
                                drift_values.append(1.0 - iou_prev)
                            prev_pred = pred_any

                        if sample_dice:
                            metric_totals["dice_video_sum"] += float(np.mean(sample_dice))
                            metric_totals["dice_video_count"] += 1.0
                        if sample_iou:
                            metric_totals["iou_video_sum"] += float(np.mean(sample_iou))
                            metric_totals["iou_video_count"] += 1.0
                        if self.main_process and sample_dice:
                            sample_name = ""
                            info = batch_data.get("info", {})
                            if isinstance(info, dict) and "name" in info:
                                name_value = info["name"]
                                sample_name = str(name_value[bi] if isinstance(name_value, (list, tuple)) and len(name_value) > bi else name_value)
                            per_video_metrics.append(
                                {
                                    "mode": mode,
                                    "iteration": int(it),
                                    "video": sample_name,
                                    "sample_index": int(bi),
                                    "dice": float(np.mean(sample_dice)),
                                    "iou": float(np.mean(sample_iou)) if sample_iou else 0.0,
                                    "ed_frame": int(ed_frame) if ed_frame is not None else "",
                                    "es_frame": int(es_frame) if es_frame is not None else "",
                                }
                            )
                        for prefix, values in (
                            ("base", base_area_values),
                            ("anchor", anchor_area_values),
                            ("proposal", proposal_area_values),
                            ("final", final_area_values),
                        ):
                            if len(values) >= 2:
                                series = torch.stack(values).float()
                                metric_totals[f"functional_anchor_{prefix}_area_range_sum"] += float(series.max().item() - series.min().item())
                                metric_totals[f"functional_anchor_{prefix}_area_std_sum"] += float(series.std(unbiased=False).item())
                                metric_totals[f"functional_anchor_{prefix}_area_count"] += 1.0
                        if len(final_area_values) >= 2:
                            final_series = torch.stack(final_area_values).float()
                            metric_totals["functional_anchor_ed_es_area_gap_sum"] += float((final_series[0] - final_series[-1]).item())
                            metric_totals["functional_anchor_ed_es_area_ratio_sum"] += float(
                                (final_series[0] / final_series[-1].clamp_min(1.0e-6)).item()
                            )
                            metric_totals["functional_anchor_ed_es_area_count"] += 1.0
                        if drift_values:
                            drift_mean = float(np.mean(drift_values))
                            metric_totals["temporal_drift_sum"] += drift_mean
                            metric_totals["temporal_drift_count"] += 1.0
                            metric_totals["temporal_dice_consistency_sum"] += 1.0 - drift_mean
                            metric_totals["temporal_dice_consistency_count"] += 1.0
                        if len(area_values) >= 3:
                            area = torch.stack(area_values).float()
                            smooth = (area[2:] - 2.0 * area[1:-1] + area[:-2]).abs().mean().item()
                            metric_totals["area_smoothness_sum"] += float(smooth)
                            metric_totals["area_smoothness_count"] += 1.0
                        if len(centroid_values) >= 2:
                            centroid = torch.stack(centroid_values).float()
                            jitter = (centroid[1:] - centroid[:-1]).pow(2).sum(dim=-1).sqrt().mean().item()
                            metric_totals["centroid_jitter_sum"] += float(jitter)
                            metric_totals["centroid_jitter_count"] += 1.0

                    if conf_pred_frames:
                        preds_concat = torch.cat(conf_pred_frames, dim=0)
                        gts_concat = torch.cat(conf_gt_frames, dim=0)
                        self.conf_metric(y_pred=preds_concat, y=gts_concat)
                        try:
                            conf_res = self.conf_metric.aggregate()
                            conf_names = ["precision", "recall", "acc", "sp", "F1"]
                            for idx, name in enumerate(conf_names):
                                metric_totals[f"{name}_sum"] += float(conf_res[idx].item())
                            metric_totals["conf_count"] += 1.0
                        except Exception:
                            pass
                        self.conf_metric.reset()

                    aux_frames = out.get("aux", []) if isinstance(out, dict) else []
                    if aux_frames:
                        for aux_t in aux_frames:
                            if not isinstance(aux_t, dict):
                                continue
                            memory_aux = aux_t.get("memory_aux", {}) if isinstance(aux_t.get("memory_aux", {}), dict) else {}
                            for name in ("gate_mean", "residual_abs_mean", "memory_update_rate", "teacher_forcing_update_prob"):
                                value = aux_t.get(name, memory_aux.get(name, None))
                                if value is None:
                                    continue
                                if torch.is_tensor(value):
                                    value = float(value.detach().float().mean().item())
                                else:
                                    value = float(value)
                                if name == "teacher_forcing_update_prob":
                                    metric_totals["teacher_forcing_prob_sum"] += value
                                else:
                                    metric_totals[f"{name}_sum"] += value
                            metric_totals["aux_count"] += 1.0

                    memory_aux_keys = sorted(k for k in out.keys() if k.startswith("memory_aux_"))
                    for key in memory_aux_keys:
                        memory_aux = out.get(key)
                        anchor_aux = memory_aux.get("anchor_ode_aux") if isinstance(memory_aux, dict) else None
                        if not isinstance(anchor_aux, dict):
                            continue
                        for src, dst in (
                            ("gate_mean", "gate_mean_sum"),
                            ("residual_abs_mean", "residual_abs_mean_sum"),
                            ("memory_update_rate", "memory_update_rate_sum"),
                        ):
                            value = anchor_aux.get(src)
                            if value is None:
                                continue
                            if torch.is_tensor(value):
                                value = float(value.detach().float().mean().item())
                            else:
                                value = float(value)
                            metric_totals[dst] += value
                        metric_totals["aux_count"] += 1.0

                    vis_limit = self.cfg.get("eval_stage", {}).get("num_vis", 0)
                    if vis_limit == 0:
                        vis_limit = self.cfg.get("num_vis", 0)

                    if self.main_process and batch_idx < vis_limit:
                        vis_path = self._visualize_batch(batch_data, out, batch_idx, it, epoch, mode)
                        if vis_path is not None:
                            if isinstance(vis_path, (list, tuple)):
                                visual_artifacts.extend(Path(path) for path in vis_path if path is not None)
                            else:
                                visual_artifacts.append(Path(vis_path))

            if "metric_totals" not in locals():
                metric_totals = {
                    "dice_frame_sum": 0.0,
                    "dice_frame_count": 0.0,
                    "iou_frame_sum": 0.0,
                    "iou_frame_count": 0.0,
                    "dice_video_sum": 0.0,
                    "dice_video_count": 0.0,
                    "iou_video_sum": 0.0,
                    "iou_video_count": 0.0,
                    "hd95_resized_sum": 0.0,
                    "hd95_resized_count": 0.0,
                    "hd95_original_sum": 0.0,
                    "hd95_original_count": 0.0,
                    "assd_resized_sum": 0.0,
                    "assd_resized_count": 0.0,
                    "assd_original_sum": 0.0,
                    "assd_original_count": 0.0,
                    "precision_sum": 0.0,
                    "recall_sum": 0.0,
                    "acc_sum": 0.0,
                    "sp_sum": 0.0,
                    "F1_sum": 0.0,
                    "conf_count": 0.0,
                    "temporal_drift_sum": 0.0,
                    "temporal_drift_count": 0.0,
                    "temporal_dice_consistency_sum": 0.0,
                    "temporal_dice_consistency_count": 0.0,
                    "area_smoothness_sum": 0.0,
                    "area_smoothness_count": 0.0,
                    "centroid_jitter_sum": 0.0,
                    "centroid_jitter_count": 0.0,
                    "gate_mean_sum": 0.0,
                    "residual_abs_mean_sum": 0.0,
                    "memory_update_rate_sum": 0.0,
                    "teacher_forcing_prob_sum": 0.0,
                    "base_only_dice_sum": 0.0,
                    "base_only_dice_count": 0.0,
                    "guided_only_dice_sum": 0.0,
                    "guided_only_dice_count": 0.0,
                    "prior_only_dice_sum": 0.0,
                    "prior_only_dice_count": 0.0,
                    "functional_anchor_base_dice_sum": 0.0,
                    "functional_anchor_base_dice_count": 0.0,
                    "functional_anchor_anchor_dice_sum": 0.0,
                    "functional_anchor_anchor_dice_count": 0.0,
                    "functional_anchor_proposal_dice_sum": 0.0,
                    "functional_anchor_proposal_dice_count": 0.0,
                    "functional_anchor_base_area_range_sum": 0.0,
                    "functional_anchor_base_area_std_sum": 0.0,
                    "functional_anchor_base_area_count": 0.0,
                    "functional_anchor_anchor_area_range_sum": 0.0,
                    "functional_anchor_anchor_area_std_sum": 0.0,
                    "functional_anchor_anchor_area_count": 0.0,
                    "functional_anchor_proposal_area_range_sum": 0.0,
                    "functional_anchor_proposal_area_std_sum": 0.0,
                    "functional_anchor_proposal_area_count": 0.0,
                    "functional_anchor_final_area_range_sum": 0.0,
                    "functional_anchor_final_area_std_sum": 0.0,
                    "functional_anchor_final_area_count": 0.0,
                    "functional_anchor_ed_es_area_gap_sum": 0.0,
                    "functional_anchor_ed_es_area_ratio_sum": 0.0,
                    "functional_anchor_ed_es_area_count": 0.0,
                    "functional_anchor_residual_l1_sum": 0.0,
                    "functional_anchor_residual_l2_sum": 0.0,
                    "functional_anchor_residual_boundary_ratio_sum": 0.0,
                    "functional_anchor_shape_residual_norm_sum": 0.0,
                    "functional_anchor_boundary_residual_norm_sum": 0.0,
                    "functional_anchor_slot_entropy_sum": 0.0,
                    "functional_anchor_ed_slot_usage_sum": 0.0,
                    "functional_anchor_early_systole_slot_usage_sum": 0.0,
                    "functional_anchor_es_slot_usage_sum": 0.0,
                    "functional_anchor_early_diastole_slot_usage_sum": 0.0,
                    "functional_anchor_uncertain_slot_usage_sum": 0.0,
                    "functional_anchor_slot_max_prob_sum": 0.0,
                    "functional_anchor_slot_area_order_violation_sum": 0.0,
                    "functional_anchor_gate_low_sum": 0.0,
                    "functional_anchor_gate_mid_sum": 0.0,
                    "functional_anchor_gate_high_sum": 0.0,
                    "functional_anchor_confidence_mean_sum": 0.0,
                    "functional_anchor_confidence_std_sum": 0.0,
                    "functional_anchor_anchor_trust_ratio_sum": 0.0,
                    "functional_anchor_image_trust_ratio_sum": 0.0,
                    "functional_anchor_trust_mean_sum": 0.0,
                    "functional_anchor_trust_std_sum": 0.0,
                    "functional_anchor_trust_spatial_std_sum": 0.0,
                    "functional_anchor_trust_disagreement_corr_sum": 0.0,
                    "functional_anchor_residual_abs_mean_sum": 0.0,
                    "functional_anchor_residual_abs_max_sum": 0.0,
                    "functional_anchor_residual_clip_hit_ratio_sum": 0.0,
                    "functional_anchor_residual_scale_sum": 0.0,
                    "functional_anchor_delta_abs_mean_sum": 0.0,
                    "functional_anchor_base_logit_abs_mean_sum": 0.0,
                    "functional_anchor_anchor_logit_abs_mean_sum": 0.0,
                    "functional_anchor_proposal_logit_abs_mean_sum": 0.0,
                    "functional_anchor_final_logit_abs_mean_sum": 0.0,
                    "functional_anchor_base_logit_std_sum": 0.0,
                    "functional_anchor_anchor_logit_std_sum": 0.0,
                    "functional_anchor_proposal_logit_std_sum": 0.0,
                    "functional_anchor_final_logit_std_sum": 0.0,
                    "functional_anchor_base_prob_mean_sum": 0.0,
                    "functional_anchor_anchor_prob_mean_sum": 0.0,
                    "functional_anchor_proposal_prob_mean_sum": 0.0,
                    "functional_anchor_final_prob_mean_sum": 0.0,
                    "functional_anchor_anchor_temperature_sum": 0.0,
                    "functional_anchor_slot_order_loss_sum": 0.0,
                    "functional_anchor_slot_area_ed_sum": 0.0,
                    "functional_anchor_slot_area_early_systole_sum": 0.0,
                    "functional_anchor_slot_area_es_sum": 0.0,
                    "functional_anchor_slot_area_early_diastole_sum": 0.0,
                    "functional_anchor_slot_area_uncertain_sum": 0.0,
                    "functional_anchor_phase_source_sum": 0.0,
                    "functional_anchor_phase_source_metadata_ratio_sum": 0.0,
                    "functional_anchor_phase_source_area_ratio_sum": 0.0,
                    "functional_anchor_phase_source_time_ratio_sum": 0.0,
                    "functional_anchor_phase_reliability_sum": 0.0,
                    "functional_anchor_phase_reliability_std_sum": 0.0,
                    "functional_anchor_phase_reliability_min_sum": 0.0,
                    "functional_anchor_phase_reliability_low_ratio_sum": 0.0,
                    "functional_anchor_state_norm_sum": 0.0,
                    "functional_anchor_state_delta_norm_sum": 0.0,
                    "functional_anchor_state_update_norm_sum": 0.0,
                    "functional_anchor_state_delta_ratio_sum": 0.0,
                    "functional_anchor_ode_raw_delta_norm_sum": 0.0,
                    "functional_anchor_ode_clamp_ratio_sum": 0.0,
                    "functional_anchor_ode_update_norm_sum": 0.0,
                    "functional_anchor_inject_gate_low_sum": 0.0,
                    "functional_anchor_inject_gate_mid_sum": 0.0,
                    "functional_anchor_inject_gate_high_sum": 0.0,
                    "functional_anchor_inject_gate_dec_sum": 0.0,
                    "functional_anchor_aux_count": 0.0,
                    "aux_count": 0.0,
                }
                for thr in threshold_candidates:
                    key = self._threshold_key(thr)
                    metric_totals[f"{key}_dice_sum"] = 0.0
                    metric_totals[f"{key}_dice_count"] = 0.0
                for key, value in self._metric_totals_template().items():
                    metric_totals.setdefault(key, value)

            global_metrics = self._reduce_metric_totals(metric_totals)
            threshold_metrics = {
                thr: global_metrics.get(f"{self._threshold_key(thr)}_dice_frame_mean", 0.0)
                for thr in threshold_candidates
            }
            if threshold_metrics:
                best_thr, best_dice = max(threshold_metrics.items(), key=lambda item: item[1])
                if mode in {"val", "validation"}:
                    self.best_val_threshold = float(best_thr)
                    self._best_val_threshold_ready = True
                    global_metrics["best_val_threshold"] = float(best_thr)
                    global_metrics["best_threshold_dice_frame_mean"] = float(best_dice)
                else:
                    global_metrics["best_val_threshold"] = float(self.best_val_threshold)
                    chosen_key = self._threshold_key(float(self.best_val_threshold))
                    global_metrics["best_threshold_dice_frame_mean"] = global_metrics.get(
                        f"{chosen_key}_dice_frame_mean", global_metrics.get("dice_frame_mean", 0.0)
                    )
                global_metrics["threshold_0p5_dice_frame_mean"] = global_metrics.get(
                    f"{self._threshold_key(0.5)}_dice_frame_mean", global_metrics.get("dice_frame_mean", 0.0)
                )

            if self.main_process:
                summary_row = self._build_summary_row(mode, global_metrics, epoch, it)
                self._append_summary_row(summary_row)

            if self.main_process:
                self._log_final_metrics(global_metrics, mode, it, epoch)

            self._save_best_if_needed(mode, global_metrics, epoch, it, raw_state=raw_state_for_restore)

            if self.is_distributed:
                dist.barrier()

            threshold_sweep = {
                f"{thr:.2f}": global_metrics.get(f"{self._threshold_key(thr)}_dice_frame_mean", 0.0)
                for thr in threshold_candidates
            }
            postprocess_cfg = self.cfg.get("evaluation", {}).get("postprocess", {})
            if hasattr(postprocess_cfg, "items"):
                postprocess = dict(postprocess_cfg.items())
            else:
                postprocess = {"enabled": bool(postprocess_cfg)}
            return EvaluationResult(
                mode=mode,
                iteration=int(it),
                epoch=int(epoch),
                summary_metrics=global_metrics,
                per_video_metrics=per_video_metrics if self.main_process else [],
                per_frame_metrics=per_frame_metrics if self.main_process else [],
                threshold_sweep=threshold_sweep,
                postprocess=postprocess,
                visual_artifacts=visual_artifacts,
            )
        finally:
            self._restore_model_state(raw_state_for_restore)
            self.model.train(prev_mode)

    def _visualize_batch(self, batch_data, out, batch_idx, it, epoch, mode):
        try:
            rgb_seq = batch_data["rgb"][0].cpu().numpy()
            cls_gt_seq = batch_data["cls_gt"][0].cpu().numpy()

            patient_name = f"b{batch_idx}"
            if "info" in batch_data and "name" in batch_data["info"]:
                patient_name = str(batch_data["info"]["name"][0])

            return visualize_sequence(
                rgb_seq,
                cls_gt_seq,
                out,
                str(self.run_path),
                f"vis_idx_{batch_idx}",
                iteration=it,
                epoch=epoch,
                patient_id=patient_name,
                mode=mode,
            )
        except Exception as e:
            self.log.warning(f"Vis failed: {e}")
            return None

    def _reduce_metric_totals(self, totals: dict):
        keys = list(totals.keys())
        vec = torch.tensor([float(totals[k]) for k in keys], device=self.device, dtype=torch.float64)
        if self.is_distributed:
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        reduced = {k: vec[idx].item() for idx, k in enumerate(keys)}

        def mean(sum_key: str, count_key: str):
            count = reduced[count_key]
            return reduced[sum_key] / count if count > 0 else 0.0

        metric_space = str(self.cfg.get("evaluation", {}).get("metric_space", "original"))
        ed_hd95 = mean("ed_hd95_original_sum", "ed_hd95_count") if metric_space == "original" else mean("ed_hd95_resized_sum", "ed_hd95_count")
        es_hd95 = mean("es_hd95_original_sum", "es_hd95_count") if metric_space == "original" else mean("es_hd95_resized_sum", "es_hd95_count")
        metrics = {
            "dice_frame_mean": mean("dice_frame_sum", "dice_frame_count"),
            "dice_video_mean": mean("dice_video_sum", "dice_video_count"),
            "iou_frame_mean": mean("iou_frame_sum", "iou_frame_count"),
            "iou_video_mean": mean("iou_video_sum", "iou_video_count"),
            "hd95_resized": mean("hd95_resized_sum", "hd95_resized_count"),
            "hd95_original": mean("hd95_original_sum", "hd95_original_count"),
            "ed_dice": mean("ed_dice_sum", "ed_dice_count"),
            "es_dice": mean("es_dice_sum", "es_dice_count"),
            "ed_hd95_resized": mean("ed_hd95_resized_sum", "ed_hd95_count"),
            "ed_hd95_original": mean("ed_hd95_original_sum", "ed_hd95_count"),
            "es_hd95_resized": mean("es_hd95_resized_sum", "es_hd95_count"),
            "es_hd95_original": mean("es_hd95_original_sum", "es_hd95_count"),
            "ed_hd95": ed_hd95,
            "es_hd95": es_hd95,
            "assd_resized": mean("assd_resized_sum", "assd_resized_count"),
            "assd_original": mean("assd_original_sum", "assd_original_count"),
            "precision": mean("precision_sum", "conf_count"),
            "recall": mean("recall_sum", "conf_count"),
            "acc": mean("acc_sum", "conf_count"),
            "sp": mean("sp_sum", "conf_count"),
            "F1": mean("F1_sum", "conf_count"),
            "temporal_drift": mean("temporal_drift_sum", "temporal_drift_count"),
            "temporal_dice_consistency": mean("temporal_dice_consistency_sum", "temporal_dice_consistency_count"),
            "area_smoothness": mean("area_smoothness_sum", "area_smoothness_count"),
            "centroid_jitter": mean("centroid_jitter_sum", "centroid_jitter_count"),
            "dice": mean("dice_frame_sum", "dice_frame_count"),
            "iou": mean("iou_frame_sum", "iou_frame_count"),
            "hd95": mean("hd95_original_sum", "hd95_original_count") if metric_space == "original" else mean("hd95_resized_sum", "hd95_resized_count"),
            "assd": mean("assd_original_sum", "assd_original_count") if metric_space == "original" else mean("assd_resized_sum", "assd_resized_count"),
            "gate_mean": mean("gate_mean_sum", "aux_count"),
            "residual_abs_mean": mean("residual_abs_mean_sum", "aux_count"),
            "memory_update_rate": mean("memory_update_rate_sum", "aux_count"),
            "teacher_forcing_prob": mean("teacher_forcing_prob_sum", "aux_count"),
            "base_only_dice_frame_mean": mean("base_only_dice_sum", "base_only_dice_count"),
            "guided_only_dice_frame_mean": mean("guided_only_dice_sum", "guided_only_dice_count"),
            "prior_only_dice_frame_mean": mean("prior_only_dice_sum", "prior_only_dice_count"),
        }
        metrics["overall_dice"] = metrics["dice"]
        metrics["overall_hd95"] = metrics["hd95"]
        if reduced.get("base_only_dice_count", 0.0) > 0:
            metrics["anchor_ode/final_dice"] = metrics["dice_frame_mean"]
            metrics["anchor_ode/base_dice"] = metrics["base_only_dice_frame_mean"]
            metrics["anchor_ode/guided_dice"] = metrics["guided_only_dice_frame_mean"]
            metrics["anchor_ode/prior_dice"] = metrics["prior_only_dice_frame_mean"]
        metrics["area_acceleration"] = metrics["area_smoothness"]
        metrics["temporal_jitter"] = metrics["temporal_drift"]
        if reduced.get("functional_anchor_aux_count", 0.0) > 0:
            metrics.update(
                {
                    "functional_anchor/base_dice": mean("functional_anchor_base_dice_sum", "functional_anchor_base_dice_count"),
                    "functional_anchor/anchor_only_dice": mean("functional_anchor_anchor_dice_sum", "functional_anchor_anchor_dice_count"),
                    "functional_anchor/proposal_dice": mean("functional_anchor_proposal_dice_sum", "functional_anchor_proposal_dice_count"),
                    "functional_anchor/base_area_range": mean("functional_anchor_base_area_range_sum", "functional_anchor_base_area_count"),
                    "functional_anchor/base_area_std": mean("functional_anchor_base_area_std_sum", "functional_anchor_base_area_count"),
                    "functional_anchor/anchor_area_range": mean("functional_anchor_anchor_area_range_sum", "functional_anchor_anchor_area_count"),
                    "functional_anchor/anchor_area_std": mean("functional_anchor_anchor_area_std_sum", "functional_anchor_anchor_area_count"),
                    "functional_anchor/proposal_area_range": mean("functional_anchor_proposal_area_range_sum", "functional_anchor_proposal_area_count"),
                    "functional_anchor/proposal_area_std": mean("functional_anchor_proposal_area_std_sum", "functional_anchor_proposal_area_count"),
                    "functional_anchor/final_area_range": mean("functional_anchor_final_area_range_sum", "functional_anchor_final_area_count"),
                    "functional_anchor/final_area_std": mean("functional_anchor_final_area_std_sum", "functional_anchor_final_area_count"),
                    "functional_anchor/ED_ES_area_gap": mean("functional_anchor_ed_es_area_gap_sum", "functional_anchor_ed_es_area_count"),
                    "functional_anchor/ED_ES_area_ratio": mean("functional_anchor_ed_es_area_ratio_sum", "functional_anchor_ed_es_area_count"),
                    "functional_anchor/residual_l1": mean("functional_anchor_residual_l1_sum", "functional_anchor_aux_count"),
                    "functional_anchor/residual_l2": mean("functional_anchor_residual_l2_sum", "functional_anchor_aux_count"),
                    "functional_anchor/residual_boundary_ratio": mean("functional_anchor_residual_boundary_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/shape_residual_norm": mean("functional_anchor_shape_residual_norm_sum", "functional_anchor_aux_count"),
                    "functional_anchor/boundary_residual_norm": mean("functional_anchor_boundary_residual_norm_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_entropy": mean("functional_anchor_slot_entropy_sum", "functional_anchor_aux_count"),
                    "functional_anchor/ED_slot_usage": mean("functional_anchor_ed_slot_usage_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_usage_ed": mean("functional_anchor_ed_slot_usage_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_usage_early_systole": mean("functional_anchor_early_systole_slot_usage_sum", "functional_anchor_aux_count"),
                    "functional_anchor/ES_slot_usage": mean("functional_anchor_es_slot_usage_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_usage_es": mean("functional_anchor_es_slot_usage_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_usage_early_diastole": mean("functional_anchor_early_diastole_slot_usage_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_usage_uncertain": mean("functional_anchor_uncertain_slot_usage_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_max_prob_mean": mean("functional_anchor_slot_max_prob_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_area_order_violation": mean("functional_anchor_slot_area_order_violation_sum", "functional_anchor_aux_count"),
                    "functional_anchor/gate_mean_low": mean("functional_anchor_gate_low_sum", "functional_anchor_aux_count"),
                    "functional_anchor/gate_mean_mid": mean("functional_anchor_gate_mid_sum", "functional_anchor_aux_count"),
                    "functional_anchor/gate_mean_high": mean("functional_anchor_gate_high_sum", "functional_anchor_aux_count"),
                    "functional_anchor/confidence_mean": mean("functional_anchor_confidence_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/confidence_std": mean("functional_anchor_confidence_std_sum", "functional_anchor_aux_count"),
                    "functional_anchor/anchor_trust_ratio": mean("functional_anchor_anchor_trust_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/image_trust_ratio": mean("functional_anchor_image_trust_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/trust_mean": mean("functional_anchor_trust_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/trust_std": mean("functional_anchor_trust_std_sum", "functional_anchor_aux_count"),
                    "functional_anchor/trust_spatial_std": mean("functional_anchor_trust_spatial_std_sum", "functional_anchor_aux_count"),
                    "functional_anchor/trust_disagreement_corr": mean("functional_anchor_trust_disagreement_corr_sum", "functional_anchor_aux_count"),
                    "functional_anchor/residual_abs_mean": mean("functional_anchor_residual_abs_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/residual_abs_max": mean("functional_anchor_residual_abs_max_sum", "functional_anchor_aux_count"),
                    "functional_anchor/residual_clip_hit_ratio": mean("functional_anchor_residual_clip_hit_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/residual_scale": mean("functional_anchor_residual_scale_sum", "functional_anchor_aux_count"),
                    "functional_anchor/delta_abs_mean": mean("functional_anchor_delta_abs_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/base_logit_abs_mean": mean("functional_anchor_base_logit_abs_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/anchor_logit_abs_mean": mean("functional_anchor_anchor_logit_abs_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/proposal_logit_abs_mean": mean("functional_anchor_proposal_logit_abs_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/final_logit_abs_mean": mean("functional_anchor_final_logit_abs_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/base_logit_std": mean("functional_anchor_base_logit_std_sum", "functional_anchor_aux_count"),
                    "functional_anchor/anchor_logit_std": mean("functional_anchor_anchor_logit_std_sum", "functional_anchor_aux_count"),
                    "functional_anchor/proposal_logit_std": mean("functional_anchor_proposal_logit_std_sum", "functional_anchor_aux_count"),
                    "functional_anchor/final_logit_std": mean("functional_anchor_final_logit_std_sum", "functional_anchor_aux_count"),
                    "functional_anchor/base_prob_mean": mean("functional_anchor_base_prob_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/anchor_prob_mean": mean("functional_anchor_anchor_prob_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/proposal_prob_mean": mean("functional_anchor_proposal_prob_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/final_prob_mean": mean("functional_anchor_final_prob_mean_sum", "functional_anchor_aux_count"),
                    "functional_anchor/anchor_temperature": mean("functional_anchor_anchor_temperature_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_order_loss": mean("functional_anchor_slot_order_loss_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_area_ed": mean("functional_anchor_slot_area_ed_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_area_early_systole": mean("functional_anchor_slot_area_early_systole_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_area_es": mean("functional_anchor_slot_area_es_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_area_early_diastole": mean("functional_anchor_slot_area_early_diastole_sum", "functional_anchor_aux_count"),
                    "functional_anchor/slot_area_uncertain": mean("functional_anchor_slot_area_uncertain_sum", "functional_anchor_aux_count"),
                    "functional_anchor/phase_source": mean("functional_anchor_phase_source_sum", "functional_anchor_aux_count"),
                    "functional_anchor/phase_source_metadata_ratio": mean("functional_anchor_phase_source_metadata_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/phase_source_area_ratio": mean("functional_anchor_phase_source_area_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/phase_source_time_ratio": mean("functional_anchor_phase_source_time_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/phase_reliability": mean("functional_anchor_phase_reliability_sum", "functional_anchor_aux_count"),
                    "functional_anchor/phase_reliability_mean": mean("functional_anchor_phase_reliability_sum", "functional_anchor_aux_count"),
                    "functional_anchor/phase_reliability_std": mean("functional_anchor_phase_reliability_std_sum", "functional_anchor_aux_count"),
                    "functional_anchor/phase_reliability_min": mean("functional_anchor_phase_reliability_min_sum", "functional_anchor_aux_count"),
                    "functional_anchor/phase_reliability_low_ratio": mean("functional_anchor_phase_reliability_low_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/state_norm": mean("functional_anchor_state_norm_sum", "functional_anchor_aux_count"),
                    "functional_anchor/state_delta_norm": mean("functional_anchor_state_delta_norm_sum", "functional_anchor_aux_count"),
                    "functional_anchor/state_update_norm": mean("functional_anchor_state_update_norm_sum", "functional_anchor_aux_count"),
                    "functional_anchor/state_delta_ratio": mean("functional_anchor_state_delta_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/ode_raw_delta_norm": mean("functional_anchor_ode_raw_delta_norm_sum", "functional_anchor_aux_count"),
                    "functional_anchor/ode_clamp_ratio": mean("functional_anchor_ode_clamp_ratio_sum", "functional_anchor_aux_count"),
                    "functional_anchor/ode_update_norm": mean("functional_anchor_ode_update_norm_sum", "functional_anchor_aux_count"),
                    "functional_anchor/inject_gate_low": mean("functional_anchor_inject_gate_low_sum", "functional_anchor_aux_count"),
                    "functional_anchor/inject_gate_mid": mean("functional_anchor_inject_gate_mid_sum", "functional_anchor_aux_count"),
                    "functional_anchor/inject_gate_high": mean("functional_anchor_inject_gate_high_sum", "functional_anchor_aux_count"),
                    "functional_anchor/inject_gate_dec": mean("functional_anchor_inject_gate_dec_sum", "functional_anchor_aux_count"),
                }
            )
            metrics["functional_anchor/final_dice"] = metrics["dice_frame_mean"]
            if reduced.get("functional_anchor_base_dice_count", 0.0) > 0:
                metrics["functional_anchor/final_minus_base"] = metrics["dice_frame_mean"] - metrics["functional_anchor/base_dice"]
            if reduced.get("functional_anchor_anchor_dice_count", 0.0) > 0:
                metrics["functional_anchor/final_minus_anchor"] = metrics["dice_frame_mean"] - metrics["functional_anchor/anchor_only_dice"]
            if reduced.get("functional_anchor_proposal_dice_count", 0.0) > 0 and reduced.get("functional_anchor_anchor_dice_count", 0.0) > 0:
                metrics["functional_anchor/proposal_minus_anchor"] = (
                    metrics["functional_anchor/proposal_dice"] - metrics["functional_anchor/anchor_only_dice"]
                )
        if reduced.get("faf_aux_count", 0.0) > 0:
            def faf_mean(sum_key: str, count_key: str = "faf_aux_count"):
                count = reduced.get(count_key, 0.0)
                return reduced.get(sum_key, 0.0) / count if count > 0 else 0.0

            metrics.update(
                {
                    "faf/base_dice": faf_mean("faf_base_dice_sum", "faf_base_dice_count"),
                    "faf/anchor_only_dice": faf_mean("faf_anchor_dice_sum", "faf_anchor_dice_count"),
                    "faf/proposal_top1_dice": faf_mean("faf_proposal_top1_dice_sum", "faf_proposal_top1_dice_count"),
                    "faf/proposal_oracle_dice": faf_mean("faf_proposal_oracle_dice_sum", "faf_proposal_dice_count"),
                    "faf/proposal_mean_dice": faf_mean("faf_proposal_mean_dice_sum", "faf_proposal_dice_count"),
                    "faf/final_dice": metrics["dice_frame_mean"],
                    "faf/effective_anchor_number": faf_mean("faf_effective_anchor_number_sum"),
                    "faf/active_anchor_entropy": faf_mean("faf_active_anchor_entropy_sum"),
                    "faf/top1_anchor_weight": faf_mean("faf_top1_anchor_weight_sum"),
                    "faf/top3_anchor_weight_sum": faf_mean("faf_top3_anchor_weight_sum"),
                    "faf/coverage_score": faf_mean("faf_coverage_score_sum"),
                    "faf/coverage_gap": faf_mean("faf_coverage_gap_sum"),
                    "faf/anchor_function_diversity": faf_mean("faf_anchor_function_diversity_sum"),
                    "faf/anchor_area_diversity": faf_mean("faf_anchor_area_diversity_sum"),
                    "faf/anchor_pairwise_similarity": faf_mean("faf_anchor_pairwise_similarity_sum"),
                    "faf/write_strength_mean": faf_mean("faf_write_strength_mean_sum"),
                    "faf/memory_update_norm": faf_mean("faf_memory_update_norm_sum"),
                    "faf/affine_delta_norm": faf_mean("faf_affine_delta_norm_sum"),
                    "faf/affine_velocity_norm": faf_mean("faf_affine_velocity_norm_sum"),
                    "faf/ode_velocity_norm": faf_mean("faf_ode_velocity_norm_sum"),
                    "faf/dead_anchor_ratio": faf_mean("faf_dead_anchor_ratio_sum"),
                    "faf/recycled_anchor_ratio": faf_mean("faf_recycled_anchor_ratio_sum"),
                    "faf/trust_mean": faf_mean("faf_trust_mean_sum"),
                    "faf/anchor_trust_ratio": faf_mean("faf_anchor_trust_ratio_sum"),
                    "faf/residual_l1": faf_mean("faf_residual_l1_sum"),
                    "faf/residual_l2": faf_mean("faf_residual_l2_sum"),
                    "faf/residual_clip_hit_ratio": faf_mean("faf_residual_clip_hit_ratio_sum"),
                    "faf/residual_scale": faf_mean("faf_residual_scale_sum"),
                    "faf/retrieval_temperature": faf_mean("faf_retrieval_temperature_sum"),
                    "faf/ode_dt": faf_mean("faf_ode_dt_sum"),
                }
            )
            if reduced.get("faf_base_dice_count", 0.0) > 0:
                metrics["faf/final_minus_base"] = metrics["dice_frame_mean"] - metrics["faf/base_dice"]
        for key in reduced:
            if key.startswith("thr_") and key.endswith("_dice_sum"):
                prefix = key[: -len("_dice_sum")]
                metrics[f"{prefix}_dice_frame_mean"] = mean(key, f"{prefix}_dice_count")
        return metrics

    def _log_final_metrics(self, metrics, mode, it, epoch):
        log_items = []
        for k, v in metrics.items():
            log_items.append(f"{k.upper()}={v:.4f}")

        log_str = f"[{mode.capitalize()}] Iter={it} | " + " | ".join(log_items)
        self.log.info(log_str)

        logger = getattr(self, "mlflow_logger", None)
        if logger is not None:
            logger.log_eval_summary(metrics, mode=mode, step=it)

    def save_weights(self, it: int):
        if not self.main_process:
            return
        self.run_path.mkdir(parents=True, exist_ok=True)
        weights_path = self.run_path / f"{self.model_name}_iter_{it}.pth"
        torch.save(self.model_without_ddp.state_dict(), weights_path)
        self.log.info(f"Saved weights: {weights_path}")
        latest_path = self.run_path / "latest.pth"
        torch.save(self.model_without_ddp.state_dict(), latest_path)
        manifest = {
            "iteration": int(it),
            "kind": "state_dict",
            "model_name": str(self.model_name),
            "stage": str(self.stage),
            "latest": latest_path.name,
            "ema_enabled": bool(self.ema is not None),
        }
        if self.ema is not None:
            ema_path = self.run_path / f"{self.model_name}_ema_iter_{it}.pth"
            torch.save(self.ema.state_dict(), ema_path)
            self.log.info(f"Saved EMA weights: {ema_path}")
            latest_ema_path = self.run_path / "latest_ema.pth"
            torch.save(self.ema.state_dict(), latest_ema_path)
            manifest["latest_ema"] = latest_ema_path.name
        manifest_path = self.run_path / "model_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        logger = getattr(self, "mlflow_logger", None)
        if logger is not None:
            logger.log_checkpoint(latest_path, artifact_name="latest.pth")
            if self.ema is not None:
                logger.log_checkpoint(self.run_path / "latest_ema.pth", artifact_name="latest_ema.pth")
            logger.log_checkpoint(manifest_path, artifact_name="model_manifest.json")

    def save_checkpoint(self, it: int):
        if not self.main_process:
            return
        self.run_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.run_path / f"{self.model_name}_{self.stage}_ckpt_{it}.pth"

        payload = {
            "it": it,
            "model": self.model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "best_val_threshold": self.best_val_threshold,
            "best_val_metric": self.best_val_metric,
        }
        torch.save(payload, ckpt_path)
        latest_path = self.run_path / "latest.pth"
        torch.save(payload, latest_path)
        logger = getattr(self, "mlflow_logger", None)
        if logger is not None:
            logger.log_checkpoint(latest_path, artifact_name="latest.pth")
        self.log.info(f"Saved checkpoint: {ckpt_path}")

    def upload_summary_artifact(self) -> None:
        if not self.main_process:
            return
        logger = getattr(self, "mlflow_logger", None)
        summary_path = self.run_path / "summary.csv"
        if logger is not None and summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if rows:
                    summary_json = self.run_path / "summary.json"
                    summary_json.write_text(json.dumps(rows[-1], indent=2, sort_keys=True), encoding="utf-8")
                    logger.log_artifact(summary_json, artifact_path="eval")
            except Exception as exc:
                self.log.warning(f"Failed to prepare summary.json artifact: {exc}")
            logger.log_artifact(summary_path, artifact_path="eval")
        if logger is not None:
            try:
                logger.log_run_logs()
            except Exception as exc:
                self.log.warning(f"Failed to upload run logs to MLflow: {exc}")

    def load_checkpoint(self, path: str):
        self.log.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model_without_ddp.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.load_state_dict(ckpt["ema"])
        if "best_val_threshold" in ckpt:
            self.best_val_threshold = float(ckpt["best_val_threshold"])
            self._best_val_threshold_ready = True
        if "best_val_metric" in ckpt:
            self.best_val_metric = float(ckpt["best_val_metric"])
        return ckpt["it"]
