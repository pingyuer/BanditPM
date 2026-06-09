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
                "stage3_num_heads": 1,
                "stage2_num_heads": 3,
                "stage3_max_offset_px": 2.0,
                "stage2_max_offset_px": 3.0,
                "padding_mode": "border",
                "align_corners": False,
                "detach_state": True,
                "stage3_decay_gate": True,
            },
        }
    )


def test_grid_anchor_router_zero_init_offsets_identity_and_positive_gamma():
    router = GridAnchorRouter(4, num_heads=3, max_offset_px=2.0)
    current = torch.randn(2, 4, 8, 8)
    out, next_anchor, aux = router(current, current.detach())
    assert out.shape == current.shape
    assert next_anchor.shape == current.shape
    assert torch.allclose(aux["offsets"], torch.zeros_like(aux["offsets"]), atol=1.0e-6)
    assert torch.allclose(out, current, atol=1.0e-6)
    assert float(aux["gamma"].item()) > 0.0
    assert aux["offset_px_mean"].shape == (2,)
    assert 0.25 < float(aux["write_mean"].mean().item()) < 0.5
    assert "head_usage_entropy" in aux
    assert aux["selector_logits"].shape == (2, 3)
    assert aux["global_selector_entropy"].shape == (2,)


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
    assert aux["stage2_offset_px_mean"].shape == (2,)
    assert aux["stage2_write_mean"].shape == (2,)
    assert aux["stage3_head_usage"].shape == (2, 1)
    assert aux["stage2_head_usage"].shape == (2, 3)
    assert aux["stage3_decay_mean"].shape == (2,)
    assert aux["stage2_selector_logit_scale"].numel() == 1
    assert aux["boundary_logits"].shape == (2, 1, 64, 64)
    assert aux["boundary_edge_gate"].shape == (2, 1, 64, 64)
    assert aux["boundary_edge_gate_mean"].shape == (2,)
    assert aux["state_detached"].item() == 1.0


def test_registry_builds_unext_gar_aliases():
    cfg = OmegaConf.create({"model": _cfg()})
    for alias in ("unext_gar", "grid_anchor_router", "gar"):
        cfg.model.name = alias
        model = MODEL_REGISTRY.build(cfg, device=torch.device("cpu"))
        assert isinstance(model, UNeXtGAR)
