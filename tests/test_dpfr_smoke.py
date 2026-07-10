import torch
from omegaconf import OmegaConf

from gdkvm_project.losses import LossComputer
from gdkvm_project.models import build_model

from tests.test_project_registries import _dpfr_cfg


def _batch(batch_size=1, frames=2, size=32):
    return {
        "rgb": torch.rand(batch_size, frames, 1, size, size),
        "cls_gt": torch.randint(0, 2, (batch_size, frames, 1, size, size)),
        "label_valid": torch.ones(batch_size, frames, dtype=torch.bool),
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
    }


def _stage_cfg():
    return OmegaConf.create(
        {
            "point_supervision": False,
            "train_num_points": 16,
            "oversample_ratio": 3.0,
            "importance_sample_ratio": 0.75,
        }
    )


def test_dpfr_forward_loss_backward_and_diagnostics():
    cfg = _dpfr_cfg()
    cfg.model.aux_loss = {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}}
    cfg.loss = {
        "name": "dpfr",
        "dpfr": {
            "lambda_final": 1.0,
            "lambda_anchor": 0.3,
            "lambda_prompt": 0.3,
            "lambda_flow_seg": 0.2,
            "lambda_flow_mag": 0.005,
            "lambda_flow_smooth": 0.01,
            "lambda_flow_temp": 0.01,
        },
    }
    model = build_model(cfg, device="cpu")
    model.train()
    data = _batch()
    data.update(model(data))
    data["supervised_indices"] = torch.ones(1, 2, dtype=torch.bool)
    assert data["logits"].shape == (1, 2, 2, 32, 32)
    assert "dpfr/fusion/prompt_gate_mean" in data["aux"]
    losses = LossComputer(cfg, _stage_cfg()).compute(data, [1])
    assert "dpfr_final" in losses
    losses["total_loss"].backward()
    assert model.prompt_encoder.transformer[0].attn.in_proj_weight.grad is not None
