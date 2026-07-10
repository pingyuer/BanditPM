from __future__ import annotations

import torch


def log_final_metrics(trainer, metrics: dict, mode: str, it: int, epoch: int) -> None:
    items = " | ".join(f"{key}={float(value):.4f}" for key, value in metrics.items() if isinstance(value, (int, float)))
    trainer.log.info(f"[{mode.capitalize()}] Iter={it} Epoch={epoch} | {items}")
    logger = getattr(trainer, "mlflow_logger", None)
    if logger is not None and hasattr(logger, "log_eval_summary"):
        logger.log_eval_summary(metrics, mode=mode, step=it)


def log_train_metrics(trainer, losses: dict, total_loss, it: int) -> None:
    payload = {
        "total_loss": float(total_loss.detach().item() if torch.is_tensor(total_loss) else total_loss),
        "lr": float(trainer.scheduler.get_last_lr()[0]),
    }
    for key, value in losses.items():
        if torch.is_tensor(value) and value.numel() == 1:
            payload[key] = float(value.detach().item())
    logger = getattr(trainer, "mlflow_logger", None)
    if logger is not None and hasattr(logger, "log_train_step"):
        logger.log_train_step(payload, step=it)
