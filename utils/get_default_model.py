"""Current-project model factory helper for smoke tests and notebooks."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from training.trainer import build_model_from_cfg


def get_default_model(
    config_name: str = "unext_fusion_echo",
    *,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Build a default model from the local Hydra configs.

    This replaces the stale CUTIE helper that referenced modules not present in
    this repository. It intentionally does not load weights.
    """

    config_dir = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(version_base="1.3.2", config_dir=str(config_dir), job_name="default_model"):
        cfg: DictConfig = compose(config_name=config_name)
    target_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_model_from_cfg(cfg, target_device).eval()
