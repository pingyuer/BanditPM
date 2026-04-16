from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import DictConfig


def seed_everything(base_seed: int, rank: int) -> None:
    seed = int(base_seed) + int(rank)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def scale_stage_for_world_size(stage_cfg: DictConfig, world_size: int) -> None:
    scale = max(int(world_size), 1)
    stage_cfg.batch_size = max(int(stage_cfg.batch_size) // scale, 1)
    stage_cfg.num_workers = max(int(stage_cfg.num_workers) // scale, 1)


def seed_dataloader_worker(_worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % (2**32))
