import os
import logging
import csv
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import wandb
from omegaconf import DictConfig

from utils.logger import TensorboardLogger
from utils.log_integrator import Integrator
from utils.time_estimator import TimeEstimator
from model.gdkvm01 import GDKVM
from model.utils.parameter_groups import get_parameter_groups
from model.losses import LossComputer
from vis.vis_0730 import visualize_sequence
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


def _contains_policy_head(name: str) -> bool:
    return (
        "prototype_manager.policy_head" in name
        or "memory_core.prototype_manager.policy_head" in name
    )

class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        stage_cfg: DictConfig,
        log: TensorboardLogger,
        run_path: str,
        train_loader,
        val_loader,
        test_loader,
    ):
        self.cfg = cfg
        self.stage_cfg = stage_cfg
        self.log = log
        self.run_path = Path(run_path)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

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

        model = GDKVM(
            use_first_frame_gt_init=bool(cfg.model.get("use_first_frame_gt_init", True)),
            prototype_value_cfg=cfg.model.get("prototype_value", None),
            temporal_memory_cfg=cfg.model.get("temporal_memory", None),
            memory_core_cfg=cfg.model.get("memory_core", None),
            use_kpff=bool(cfg.model.get("use_kpff", True)),
        ).to(self.device)
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
                wandb.watch(self.model, log="all", log_freq=100)
            except Exception:
                pass

            try:
                param_count = sum(p.nelement() for p in self.model.parameters()) / 1e6
                self.log.info(f"Model Parameters: {param_count:.2f}M")
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
        default_mode = "oracle_gt" if bool(self.cfg.model.get("use_first_frame_gt_init", True)) else "pred_or_zero"
        return str(phase_cfg.get(phase, self.cfg.get("evaluation", {}).get("init_mode", default_mode)))

    def _resolve_eval_indices(self, data):
        T = data["rgb"].shape[1]
        frame_scope = str(self.cfg.get("evaluation", {}).get("frame_scope", "supervised_only"))
        if frame_scope == "all_available":
            eval_valid = data.get("eval_valid")
            if eval_valid is not None:
                source = eval_valid
            else:
                source = data.get("label_valid")
        else:
            source = data.get("label_valid")

        if source is None:
            return torch.ones((data["rgb"].shape[0], T), device=self.device, dtype=torch.bool)

        frame_mask = self._resolve_frame_valid_mask(source, batch_size=data["rgb"].shape[0], total_frames=T)
        if not frame_mask.any():
            return self._resolve_supervised_indices(data)
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
        return {
            "mode": mode,
            "iteration": it,
            "epoch": epoch,
            "experiment_name": self.exp_id,
            "dataset": str(self.cfg.get("dataset_name", "")),
            "protocol_name": str(self.cfg.get("data", {}).get("protocol_name", "unknown")),
            "init_mode": init_mode,
            "frame_scope": str(self.cfg.get("evaluation", {}).get("frame_scope", "supervised_only")),
            "metric_space": metric_space,
            "dice_frame_mean": metrics.get("dice_frame_mean", 0.0),
            "dice_video_mean": metrics.get("dice_video_mean", 0.0),
            "iou_frame_mean": metrics.get("iou_frame_mean", 0.0),
            "iou_video_mean": metrics.get("iou_video_mean", 0.0),
            "hd95_resized": metrics.get("hd95_resized", 0.0),
            "hd95_original": metrics.get("hd95_original", 0.0),
            "assd_resized": metrics.get("assd_resized", 0.0),
            "assd_original": metrics.get("assd_original", 0.0),
            "temporal_drift": metrics.get("temporal_drift", 0.0),
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

        if it % self.log_text_interval == 0:
            loss_val = loss.item()
            self.log.log_scalar("loss", loss_val, it)

            if self.main_process:
                self._wandb_log(losses, loss_val, it)
                if it != 0:
                    self.log.log_scalar("lr", self.scheduler.get_last_lr()[0], it)
                self._log_bpm_stats(data, it)
                self._log_dynakey_stats(data, it)

        return loss.detach()

    def _wandb_log(self, losses, total_loss, it):
        try:
            log_dict = {
                "train/loss": total_loss,
                "train/lr": self.scheduler.get_last_lr()[0],
            }
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    log_dict[f"train/{k}"] = v.item()
            wandb.log(log_dict, step=it)
        except Exception:
            pass

    def _log_bpm_stats(self, data, it: int) -> None:
        bpm_keys = sorted(k for k in data.keys() if k.startswith("bpm_aux_"))
        if not bpm_keys:
            return

        try:
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
                    self.log.log_scalar(f"bpm/action_{name}", hist[idx].item(), it)

            if age_tensors:
                self.log.log_scalar("bpm/age_mean", torch.cat(age_tensors).mean().item(), it)
            if usage_tensors:
                self.log.log_scalar("bpm/usage_mean", torch.cat(usage_tensors).mean().item(), it)
            if conf_tensors:
                self.log.log_scalar("bpm/conf_mean", torch.cat(conf_tensors).mean().item(), it)
            if entropy_tensors:
                self.log.log_scalar("bpm/policy_entropy", torch.cat(entropy_tensors).mean().item(), it)
            if agreement_tensors:
                self.log.log_scalar("bpm/rule_learned_agreement", torch.cat(agreement_tensors).mean().item(), it)
            if occupancy_tensors:
                self.log.log_scalar("bpm/occupancy_ratio", torch.cat(occupancy_tensors).mean().item(), it)
        except Exception:
            pass

    def _log_dynakey_stats(self, data, it: int) -> None:
        memory_keys = sorted(k for k in data.keys() if k.startswith("memory_aux_"))
        if not memory_keys:
            return

        try:
            occupancy_tensors = []
            active_count_tensors = []
            entropy_tensors = []
            fallback_tensors = []
            prediction_error_tensors = []
            residual_tensors = []
            action_hist_tensors = []
            for key in memory_keys:
                aux = data[key]
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

            if occupancy_tensors:
                self.log.log_scalar("dynakey/occupancy_ratio", torch.cat(occupancy_tensors).mean().item(), it)
            if active_count_tensors:
                self.log.log_scalar("dynakey/active_key_count", torch.cat(active_count_tensors).mean().item(), it)
            if entropy_tensors:
                self.log.log_scalar("dynakey/retrieval_entropy", torch.cat(entropy_tensors).mean().item(), it)
            if fallback_tensors:
                self.log.log_scalar("dynakey/identity_fallback", torch.cat(fallback_tensors).mean().item(), it)
            if prediction_error_tensors:
                self.log.log_scalar("dynakey/prediction_error", torch.cat(prediction_error_tensors).mean().item(), it)
            if residual_tensors:
                self.log.log_scalar("dynakey/residual_norm", torch.cat(residual_tensors).mean().item(), it)
            if action_hist_tensors:
                hist = torch.stack(action_hist_tensors, dim=0).mean(dim=0)
                names = ["keep", "update", "spawn", "split", "delete"]
                for idx, name in enumerate(names):
                    self.log.log_scalar(f"dynakey/action_{name}", hist[idx].item(), it)
        except Exception:
            pass

    def evaluate(
        self, val_loader, epoch, run_path, it, local_rank=None, world_size=None
    ):
        return self._run_evaluation(val_loader, "val", epoch, run_path, it)

    def test(self, test_loader, epoch, run_path, it, local_rank=None, world_size=None):
        return self._run_evaluation(test_loader, "test", epoch, run_path, it)

    def _reset_metrics(self):
        self.conf_metric.reset()

    def _run_evaluation(self, data_loader, mode, epoch, run_path, it):
        if self.is_distributed:
            dist.barrier()

        if self.main_process:
            self.log.info(
                f"[{mode.capitalize()}] Iter {it} Epoch {epoch}: Start Evaluation..."
            )

        prev_mode = self.model.training
        self.model.eval()
        self._reset_metrics()

        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                self._move_to_device(batch_data)
                batch_data["init_mode"] = self._resolve_phase_init(mode)

                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    out = self.model(batch_data)
                    out = self._ensure_finite_outputs(out)

                supervised_indices = self._resolve_supervised_indices(batch_data)
                eval_indices = self._resolve_eval_indices(batch_data)
                required_eval_ids = sorted(torch.nonzero(eval_indices.any(dim=0), as_tuple=False).flatten().tolist())
                mask_keys = [f"masks_{ti}" for ti in required_eval_ids]

                if batch_idx == 0 and self.main_process:
                    self.log.info(
                        f"[{mode.capitalize()}] init_mode={batch_data['init_mode']} | "
                        f"metric_space={str(self.cfg.get('evaluation', {}).get('metric_space', 'original'))} | "
                        f"supervised_indices={self._format_frame_mask(supervised_indices)} | "
                        f"eval_indices={self._format_frame_mask(eval_indices)}"
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
                    }

                conf_pred_frames = []
                conf_gt_frames = []

                for bi in range(batch_size):
                    sample_dice = []
                    sample_iou = []
                    drift_values = []
                    original_sizes = batch_data.get("original_size")
                    sample_eval_indices = torch.nonzero(eval_indices[bi], as_tuple=False).flatten().tolist()
                    for ti in sample_eval_indices:
                        pred = out[f"masks_{ti}"][bi:bi + 1]
                        if pred.shape[1] > 1:
                            pred = pred[:, 1:2, ...]
                        pred_bin = (pred > 0.5).float()
                        gt_frame = gt[bi, ti, ...].unsqueeze(0).unsqueeze(0).float()

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

                        conf_pred_frames.append(pred_bin)
                        conf_gt_frames.append(gt_frame)

                    all_mask_keys = sorted(
                        (key for key in out.keys() if key.startswith("masks_")),
                        key=lambda key: int(key.split("_")[-1]),
                    )
                    prev_pred = None
                    for key in all_mask_keys:
                        pred_any = out[key][bi:bi + 1]
                        if pred_any.shape[1] > 1:
                            pred_any = pred_any[:, 1:2, ...]
                        pred_any = (pred_any > 0.5).float()
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
                    if drift_values:
                        metric_totals["temporal_drift_sum"] += float(np.mean(drift_values))
                        metric_totals["temporal_drift_count"] += 1.0

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

                vis_limit = self.cfg.get("eval_stage", {}).get("num_vis", 0)
                if vis_limit == 0:
                    vis_limit = self.cfg.get("num_vis", 0)

                if self.main_process and batch_idx < vis_limit:
                    self._visualize_batch(batch_data, out, batch_idx, it, epoch, mode)

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
            }

        global_metrics = self._reduce_metric_totals(metric_totals)

        if self.main_process:
            summary_row = self._build_summary_row(mode, global_metrics, epoch, it)
            self._append_summary_row(summary_row)

        if self.main_process:
            self._log_final_metrics(global_metrics, mode, it, epoch)

        if self.is_distributed:
            dist.barrier()

        self.model.train(prev_mode)
        return global_metrics

    def _visualize_batch(self, batch_data, out, batch_idx, it, epoch, mode):
        try:
            rgb_seq = batch_data["rgb"][0].cpu().numpy()
            cls_gt_seq = batch_data["cls_gt"][0].cpu().numpy()

            patient_name = f"b{batch_idx}"
            if "info" in batch_data and "name" in batch_data["info"]:
                patient_name = str(batch_data["info"]["name"][0])

            visualize_sequence(
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
        return {
            "dice_frame_mean": mean("dice_frame_sum", "dice_frame_count"),
            "dice_video_mean": mean("dice_video_sum", "dice_video_count"),
            "iou_frame_mean": mean("iou_frame_sum", "iou_frame_count"),
            "iou_video_mean": mean("iou_video_sum", "iou_video_count"),
            "hd95_resized": mean("hd95_resized_sum", "hd95_resized_count"),
            "hd95_original": mean("hd95_original_sum", "hd95_original_count"),
            "assd_resized": mean("assd_resized_sum", "assd_resized_count"),
            "assd_original": mean("assd_original_sum", "assd_original_count"),
            "precision": mean("precision_sum", "conf_count"),
            "recall": mean("recall_sum", "conf_count"),
            "acc": mean("acc_sum", "conf_count"),
            "sp": mean("sp_sum", "conf_count"),
            "F1": mean("F1_sum", "conf_count"),
            "temporal_drift": mean("temporal_drift_sum", "temporal_drift_count"),
            "dice": mean("dice_frame_sum", "dice_frame_count"),
            "iou": mean("iou_frame_sum", "iou_frame_count"),
            "hd95": mean("hd95_original_sum", "hd95_original_count") if metric_space == "original" else mean("hd95_resized_sum", "hd95_resized_count"),
            "assd": mean("assd_original_sum", "assd_original_count") if metric_space == "original" else mean("assd_resized_sum", "assd_resized_count"),
        }

    def _log_final_metrics(self, metrics, mode, it, epoch):
        log_items = []
        for k, v in metrics.items():
            log_items.append(f"{k.upper()}={v:.4f}")
        
        log_str = f"[{mode.capitalize()}] Iter={it} | " + " | ".join(log_items)
        self.log.info(log_str)

        try:
            w_metrics = {f"{mode}/{k}": v for k, v in metrics.items()}
            w_metrics["epoch"] = epoch
            wandb.log(w_metrics, step=it)
        except Exception:
            pass

    def save_weights(self, it: int):
        if not self.main_process:
            return
        self.run_path.mkdir(parents=True, exist_ok=True)
        weights_path = self.run_path / f"{self.model_name}_iter_{it}.pth"
        torch.save(self.model_without_ddp.state_dict(), weights_path)
        self.log.info(f"Saved weights: {weights_path}")

    def save_checkpoint(self, it: int):
        if not self.main_process:
            return
        self.run_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.run_path / f"{self.model_name}_{self.stage}_ckpt_{it}.pth"

        torch.save(
            {
                "it": it,
                "model": self.model_without_ddp.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            ckpt_path,
        )
        self.log.info(f"Saved checkpoint: {ckpt_path}")

    def load_checkpoint(self, path: str):
        self.log.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model_without_ddp.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        return ckpt["it"]
