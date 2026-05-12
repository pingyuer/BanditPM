from __future__ import annotations

import os

from utils.registry import Registry
from dataset.echo import EchoDataset
from dataset.vos_dataset import TenCamusDataset


DATASET_REGISTRY = Registry("dataset")
DATASET_REGISTRY.register("echo", module=EchoDataset)
DATASET_REGISTRY.register("echonet", module=EchoDataset)
DATASET_REGISTRY.register("domain", module=EchoDataset)
DATASET_REGISTRY.register("cardiacuda", module=EchoDataset)
DATASET_REGISTRY.register("camus", module=TenCamusDataset)


def infer_dataset_name(cfg) -> str:
    dataset_name = str(cfg.get("dataset_name", "")).lower().strip()
    if dataset_name:
        return dataset_name

    data_path = os.path.expanduser(str(cfg.get("data_path", ""))).lower()
    if "cardiacuda" in data_path:
        return "cardiacuda"
    if "echonet" in data_path or "echo" in data_path:
        return "echonet"
    return "camus"


def resolve_dataset_class_from_cfg(cfg):
    dataset_name = infer_dataset_name(cfg)
    return dataset_name, DATASET_REGISTRY.get(dataset_name)
