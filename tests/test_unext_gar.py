from __future__ import annotations

from omegaconf import OmegaConf
import torch

from model.grid_anchor_router import GridAnchorRouter
from model.unext_gar import UNeXtGAR
from models.registry import MODEL_REGISTRY


def _cfg():
    return OmegaConf.create(
        {
            "name": "unext_gar",
            "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
            "temporal_memory": {"type": "none", "bpm": {"ENABLE": False}},
            "unext_gar": {
                "in_channels": 1,
                "num_classes": 2,
                "base_dim": 8,
                "value_dim": 16,
                "num_heads": 3,
                "max_offset": 0.1,
                "padding_mode": "border",
                "align_corners": False,
                "detach_state": True,
            },
        }
    )


def test_grid_anchor_router_zero_init_offsets_identity_and_gamma_zero():
    router = GridAnchorRouter(4, num_heads=3, max_offset=0.1)
    current = torch.randn(2, 4, 8, 8)
    out, next_anchor, aux = router(current, current.detach())
    assert out.shape == current.shape
    assert next_anchor.shape == current.shape
    assert torch.allclose(aux["offsets"], torch.zeros_like(aux["offsets"]), atol=1.0e-6)
    assert torch.allclose(out, current, atol=1.0e-6)
    assert float(aux["gamma"].item()) == 0.0


def test_unext_gar_forward_contract_and_aux_shapes():
    model = UNeXtGAR(_cfg())
    data = {
        "rgb": torch.rand(2, 3, 1, 64, 64),
        "info": {"num_objects": torch.ones(2, dtype=torch.long)},
    }
    out = model(data)
    assert out["masks_0"].shape == (2, 1, 64, 64)
    assert out["logits_2"].shape == (2, 2, 64, 64)
    aux = out["memory_aux_1"]["gar_aux"]
    assert aux["proposal_logits"].shape == (2, 1, 3, 64, 64)
    assert aux["head_weights"].shape == (2, 1, 3)
    assert aux["stage2_offset_abs_mean"].shape == (2,)
    assert aux["state_detached"].item() == 1.0


def test_registry_builds_unext_gar_aliases():
    cfg = OmegaConf.create({"model": _cfg()})
    for alias in ("unext_gar", "grid_anchor_router", "gar"):
        cfg.model.name = alias
        model = MODEL_REGISTRY.build(cfg, device=torch.device("cpu"))
        assert isinstance(model, UNeXtGAR)
