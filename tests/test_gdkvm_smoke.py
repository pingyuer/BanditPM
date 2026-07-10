import torch
from omegaconf import OmegaConf

from gdkvm_project.losses import LossComputer
from gdkvm_project.models import build_model
from tests.test_project_registries import _gdkvm_cfg


def test_gdkvm_forward_and_segmentation_loss_smoke():
    cfg = _gdkvm_cfg()
    cfg.model.aux_loss = {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}}
    model = build_model(cfg, device="cpu")
    model.eval()
    data = {
        "rgb": torch.rand(1, 2, 1, 32, 32),
        "ff_gt": torch.randint(0, 2, (1, 1, 1, 32, 32)).float(),
        "cls_gt": torch.randint(0, 2, (1, 2, 1, 32, 32)),
        "info": {"num_objects": torch.ones(1, dtype=torch.long)},
        "init_mode": "pred_or_zero",
        "supervised_indices": torch.ones(1, 2, dtype=torch.bool),
    }
    with torch.no_grad():
        data.update(model(data))
    data["aux_0"] = {}
    data["aux_1"] = {}
    assert data["logits_0"].shape == (1, 2, 32, 32)
    assert data["masks_1"].shape == (1, 1, 32, 32)
    stage_cfg = OmegaConf.create(
        {
            "point_supervision": False,
            "train_num_points": 16,
            "oversample_ratio": 3.0,
            "importance_sample_ratio": 0.75,
        }
    )
    losses = LossComputer(cfg, stage_cfg).compute(data, [1])
    assert losses["total_loss"].item() > 0.0
