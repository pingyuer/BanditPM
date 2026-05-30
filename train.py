import os
import math
import logging
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

from utils.ddp import distributed_setup, info_if_rank_zero, is_main_process, barrier

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from dataset.registry import resolve_dataset_class_from_cfg
from training import Trainer, TrainingLogger
from experiment import MLflowLogger
from utils.training_setup import (
    scale_stage_for_world_size,
    seed_dataloader_worker,
    seed_everything,
)

log = logging.getLogger(__name__)

from experiment.metadata import (
    build_mlflow_metadata,
    resolve_git_metadata,
    resolve_git_short_hash,
    resolve_mlflow_experiment_name,
    resolve_mlflow_run_name,
    resolve_model_name,
)


def resolve_dataset_class(cfg: DictConfig):
    """Resolve the dataset class through the dataset registry."""
    return resolve_dataset_class_from_cfg(cfg)


def _tensor_unique_preview(tensor: torch.Tensor, limit: int = 16) -> list:
    values = torch.unique(tensor.detach().cpu())
    preview = values[:limit].tolist()
    return [int(v) if float(v).is_integer() else float(v) for v in preview]


def _label_valid_hist(label_valid: torch.Tensor | None) -> dict[str, int]:
    if label_valid is None:
        return {}
    counts = label_valid.detach().cpu().bool().sum(dim=1).tolist()
    hist = Counter(int(v) for v in counts)
    return {str(k): int(v) for k, v in sorted(hist.items())}


def _probe_dataset_batch(loader, batch_size: int) -> dict:
    dataset = getattr(loader, "dataset", None)
    if dataset is None or len(dataset) == 0:
        return {"sample_count": 0}
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    python_state = random.getstate()
    probe_batch_size = max(1, min(int(batch_size), len(dataset)))
    try:
        probe_loader = data.DataLoader(dataset, batch_size=probe_batch_size, shuffle=False, num_workers=0)
        batch = next(iter(probe_loader))
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        random.setstate(python_state)
    rgb = batch["rgb"]
    cls_gt = batch["cls_gt"]
    label_valid = batch.get("label_valid")
    return {
        "sample_count": len(dataset),
        "rgb_shape": list(rgb.shape),
        "cls_gt_shape": list(cls_gt.shape),
        "rgb_min": float(rgb.min().item()),
        "rgb_max": float(rgb.max().item()),
        "cls_gt_unique": _tensor_unique_preview(cls_gt),
        "label_valid_hist": _label_valid_hist(label_valid),
    }


def _log_data_flow_summary(
    *,
    cfg: DictConfig,
    dataset_name: str,
    stage_cfg: DictConfig,
    train_loader,
    val_loader,
    test_loader,
    run_dir: str,
    mlflow_logger: MLflowLogger,
    main_process: bool,
) -> None:
    if not main_process:
        return
    data_path = os.path.expanduser(str(cfg.data_path))
    train_probe = _probe_dataset_batch(train_loader, int(stage_cfg.batch_size))
    val_probe = {"sample_count": len(getattr(val_loader, "dataset", []))}
    test_probe = {"sample_count": len(getattr(test_loader, "dataset", []))}
    summary = {
        "dataset": dataset_name,
        "data_path": data_path,
        "processed_root": str(cfg.get("processed_root", Path(data_path).parent)),
        "seq_length": int(stage_cfg.seq_length),
        "crop_size": list(stage_cfg.crop_size),
        "splits": {
            "train": train_probe.get("sample_count", 0),
            "val": val_probe.get("sample_count", 0),
            "test": test_probe.get("sample_count", 0),
        },
        "first_train_batch": train_probe,
        "evaluation": {
            "frame_scope": str(cfg.get("evaluation", {}).get("frame_scope", "supervised_only")),
            "exclude_init_frame": bool(cfg.get("evaluation", {}).get("exclude_init_frame", False)),
            "eval_protocol": str(cfg.get("evaluation", {}).get("eval_protocol", "")),
        },
    }
    info_if_rank_zero("[DataFlow] " + json.dumps(summary, sort_keys=True, default=str))
    params = {
        "data/name": dataset_name,
        "data/path": data_path,
        "data/seq_length": summary["seq_length"],
        "data/crop_size": summary["crop_size"],
        "data/split_count/train": summary["splits"]["train"],
        "data/split_count/val": summary["splits"]["val"],
        "data/split_count/test": summary["splits"]["test"],
        "data/mask_unique_sample": train_probe.get("cls_gt_unique", []),
        "data/label_valid_hist_sample": train_probe.get("label_valid_hist", {}),
        "eval/frame_scope": summary["evaluation"]["frame_scope"],
        "eval/exclude_init_frame": summary["evaluation"]["exclude_init_frame"],
        "eval/protocol": summary["evaluation"]["eval_protocol"],
    }
    mlflow_logger.log_params(params)
    path = Path(run_dir) / "data_flow_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    mlflow_logger.log_artifact(path, artifact_path="data")


@hydra.main(version_base="1.3.2", config_path="config", config_name="config_banditpm_baseline.yaml")
def train(cfg: DictConfig):
    dataset_name, dataset_cls = resolve_dataset_class(cfg)

    # -------- DDP Initialization --------
    local_rank, world_size = distributed_setup(backend="nccl")
    main_process = is_main_process()
    run_dir = HydraConfig.get().run.dir
    mlflow_cfg = cfg.get("mlflow", {})
    if hasattr(mlflow_cfg, "get") and not mlflow_cfg.get("experiment_name", None):
        mlflow_cfg.experiment_name = resolve_mlflow_experiment_name(cfg)
    if hasattr(mlflow_cfg, "get") and not mlflow_cfg.get("run_name", None):
        mlflow_cfg.run_name = resolve_mlflow_run_name(cfg)
    mlflow_logger = MLflowLogger(
        mlflow_cfg,
        run_dir=run_dir,
        enabled=bool(mlflow_cfg.get("enabled", True)) if hasattr(mlflow_cfg, "get") else True,
        main_process=main_process,
    )
    mlflow_started = False
    trainer = None

    try:
        stage = str(mlflow_cfg.get("stage", "full")) if hasattr(mlflow_cfg, "get") else "full"
        mlflow_enabled = bool(mlflow_cfg.get("enabled", True)) if hasattr(mlflow_cfg, "get") else True
        mlflow_required = bool(mlflow_cfg.get("required", True)) if hasattr(mlflow_cfg, "get") else True
        if main_process and stage in {"full", "final"} and mlflow_required and not mlflow_enabled:
            raise RuntimeError("Formal full/final runs require mlflow.enabled=true.")
        mlflow_preflight = bool(mlflow_cfg.get("preflight", False)) if hasattr(mlflow_cfg, "get") else False
        if main_process and mlflow_enabled and mlflow_required and mlflow_preflight and stage in {"full", "final"}:
            info_if_rank_zero("MLflow: preflight...")
            mlflow_logger.preflight()
        info_if_rank_zero("MLflow: starting run...")
        mlflow_logger.start_run()
        mlflow_started = True
        info_if_rank_zero("MLflow: logging run metadata...")
        metadata_tags, metadata_params = build_mlflow_metadata(cfg, world_size=world_size)
        mlflow_logger.log_run_metadata(tags=metadata_tags, params=metadata_params)
        info_if_rank_zero("MLflow: logging config...")
        mlflow_logger.log_config(cfg, overrides=list(HydraConfig.get().overrides.task))
        info_if_rank_zero("MLflow: logging environment info...")
        mlflow_logger.log_env_info()
        info_if_rank_zero("MLflow: logging git info...")
        mlflow_logger.log_git_info()
        info_if_rank_zero("MLflow: run initialization complete.")

        # Ensure configuration is printed only once by the main process
            
        info_if_rank_zero(f"All configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
        info_if_rank_zero(f"Number of detected GPUs: {world_size}")
        info_if_rank_zero(f"Run dir: {run_dir}")
        info_if_rank_zero(f"Dataset: {dataset_name}")

        if cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # -------- Random Seed (Offset by rank to avoid identical augmentations) --------
        base_seed = int(cfg.seed)
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed_everything(base_seed, rank)

        # -------- Adjust per-GPU batch size and workers based on world_size --------
        stage_cfg = cfg.main_training
        info_if_rank_zero(f"batch_size={stage_cfg.batch_size}")
        original_num_workers = int(stage_cfg.num_workers)
        scale_stage_for_world_size(stage_cfg, world_size)
        info_if_rank_zero(f"batch_size(per-GPU)={stage_cfg.batch_size}")

        info_if_rank_zero(f"num_workers={original_num_workers}")
        info_if_rank_zero(f"num_workers(per-GPU)={stage_cfg.num_workers}")

        # -------- Logging: Only main process writes to MLflow --------
        log_writer = TrainingLogger(run_dir, logging.getLogger())

        # -------- DataLoader Factory (Robust with fallback) --------
        def create_safe_dataloader(
            dataset,
            sampler,
            batch_size,
            num_workers,
            *,
            pin_memory=True,
            drop_last=False,
        ):
            try:
                return data.DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                    shuffle=False,
                    drop_last=drop_last,
                    worker_init_fn=seed_dataloader_worker,
                )
            except Exception as e:
                if main_process:
                    log.warning(
                        f"DataLoader failed with num_workers={num_workers}, fallback to 0: {e}"
                    )
                return data.DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=0,
                    pin_memory=False,
                    shuffle=False,
                    drop_last=drop_last,
                )

        def build_loader(mode, *, shuffle, drop_last):
            data_cfg = cfg.get("data", {})
            dataset = dataset_cls(
                filepath=os.path.expanduser(str(cfg.data_path)),
                mode=mode,
                seq_length=stage_cfg.seq_length,
                max_num_obj=stage_cfg.num_objects,
                size=stage_cfg.crop_size[0],
                augmentation=cfg.get("augmentation", {}) if mode == "train" else {},
                lv_class_id=data_cfg.get("lv_class_id", None) if hasattr(data_cfg, "get") else None,
            )
            if world_size > 1 and dist.is_initialized():
                sampler = DistributedSampler(
                    dataset,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )
            else:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            loader = create_safe_dataloader(
                dataset=dataset,
                sampler=sampler,
                batch_size=stage_cfg.batch_size,
                num_workers=stage_cfg.num_workers,
                pin_memory=True,
                drop_last=drop_last,
            )
            return loader, sampler

        # -------- Dataset / Sampler / DataLoader --------
        train_loader, train_sampler = build_loader("train", shuffle=True, drop_last=True)
        val_loader, val_sampler = build_loader("val", shuffle=False, drop_last=False)
        test_loader, test_sampler = build_loader("test", shuffle=False, drop_last=False)

        _log_data_flow_summary(
            cfg=cfg,
            dataset_name=dataset_name,
            stage_cfg=stage_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_dir=run_dir,
            mlflow_logger=mlflow_logger,
            main_process=main_process,
        )

        # -------- Trainer --------
        trainer = Trainer(
            cfg=cfg,
            stage_cfg=stage_cfg,
            log=log_writer,
            run_path=run_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            mlflow_logger=mlflow_logger,
        )

        total_iterations = int(stage_cfg.num_iterations)
        steps_per_epoch = len(train_loader)
        max_epoch = math.ceil(total_iterations / max(steps_per_epoch, 1))
        info_if_rank_zero(f"Total iterations: {total_iterations}")
        info_if_rank_zero(f"train_loader length (batches per epoch): {steps_per_epoch}")
        info_if_rank_zero(f"Total iteration={total_iterations}, est. epochs={max_epoch} ...")

        # Display progress bar only on main process
        pbar = tqdm(total=total_iterations, desc="Training", ncols=120) if main_process else None

        save_enabled = getattr(cfg, "save", 0) == 1
        weights_interval = getattr(cfg, "save_weights_interval", 0)
        checkpoint_interval = getattr(cfg, "save_checkpoint_interval", 0)
        eval_interval = getattr(cfg.eval_stage, "eval_interval", 0)

        # -------- Iteration-based Training Loop --------
        data_iter = None
        for it in range(total_iterations):
            epoch = it // steps_per_epoch
            # Set seed and rebuild iterator at the start of each epoch
            if it % steps_per_epoch == 0:
                if isinstance(train_sampler, DistributedSampler):
                    train_sampler.set_epoch(epoch)
                data_iter = iter(train_loader)

            try:
                batch_data = next(data_iter)
            except StopIteration:
                # Fallback safety mechanism (rarely triggered)
                if isinstance(train_sampler, DistributedSampler):
                    train_sampler.set_epoch(epoch)
                data_iter = iter(train_loader)
                batch_data = next(data_iter)

            loss_val = trainer.do_pass(batch_data, it)

            # ----- Save (Main process only), barrier used to prevent concurrency issues -----
            if save_enabled and it > 0:
                if weights_interval and it % weights_interval == 0:
                    if main_process:
                        trainer.save_weights(it)
                        log.info(f"Weights saved at iteration {it} (epoch {epoch+1})")
                    barrier()

                if checkpoint_interval and it % checkpoint_interval == 0:
                    if main_process:
                        trainer.save_checkpoint(it)
                        log.info(f"Checkpoint saved at iteration {it} (epoch {epoch+1})")
                    barrier()

            # Update progress bar only on main process
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"iter": it + 1, "loss": f"{loss_val:.4f}"})

            # ----- Periodic Evaluation/Testing: Fix eval shard before evaluation -----
            is_final_iter = it + 1 == total_iterations
            final_eval = bool(getattr(cfg.eval_stage, "final_eval", True))
            periodic_eval = bool(eval_interval and (it + 1) % eval_interval == 0)
            should_eval = periodic_eval or (final_eval and is_final_iter)
            full_eval = bool(final_eval and is_final_iter)
            if should_eval:
                if pbar is not None:
                    print()
                eval_seed = epoch  # Or fixed to 0
                if isinstance(val_sampler, DistributedSampler):
                    val_sampler.set_epoch(eval_seed)
                if isinstance(test_sampler, DistributedSampler):
                    test_sampler.set_epoch(eval_seed)

                trainer.evaluate(
                    val_loader=val_loader,
                    epoch=epoch + 1,
                    local_rank=local_rank,
                    world_size=world_size,
                    run_path=run_dir,
                    it=it + 1,
                    full_eval=full_eval,
                )

                test_interval = int(getattr(cfg.eval_stage, "test_interval", 0) or 0)
                test_every_eval = bool(getattr(cfg.eval_stage, "test_every_eval", False))
                final_test = bool(getattr(cfg.eval_stage, "final_test", True))
                should_test = (
                    test_every_eval
                    or bool(test_interval and (it + 1) % test_interval == 0)
                    or bool(full_eval and final_test)
                )
                if should_test:
                    trainer.test(
                        test_loader=test_loader,
                        epoch=epoch + 1,
                        local_rank=local_rank,
                        world_size=world_size,
                        run_path=run_dir,
                        it=it + 1,
                        full_eval=full_eval,
                    )

        info_if_rank_zero("Training completed.")
        if trainer is not None:
            trainer.upload_summary_artifact()
        if mlflow_started:
            mlflow_logger.end_run()

    except Exception:
        if mlflow_started:
            mlflow_logger.mark_failed()
        raise

    finally:
        # Synchronize all processes before closing resources
        barrier()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    train()
