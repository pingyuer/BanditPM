from __future__ import annotations

from omegaconf import OmegaConf
import torch

from model.cardia import CARDIA, GridODESolver
from models.registry import MODEL_REGISTRY


def _cfg():
    return OmegaConf.create(
        {
            "name": "cardia",
            "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
            "temporal_memory": {"type": "none", "bpm": {"ENABLE": False}},
            "cardia": {
                "in_channels": 1,
                "num_classes": 2,
                "base_dim": 8,
                "value_dim": 16,
                "stage3_num_heads": 1,
                "stage2_num_heads": 3,
                "stage3_max_offset_px": 1.5,
                "stage2_max_offset_px": 3.0,
                "padding_mode": "border",
                "align_corners": False,
                "detach_runtime_state": True,
                "stage3_decay_gate": True,
            },
        }
    )


def test_grid_ode_solver_identity_samples_current_anchor():
    solver = GridODESolver(padding_mode="border", align_corners=False)
    anchor_feat_t = torch.randn(2, 4, 8, 8)
    ode_flow_t = torch.zeros(2, 3, 2, 8, 8)
    selector = torch.zeros(2, 3, 8, 8)
    selector[:, 0] = 1.0
    dynamic_anchor_t, solved = solver(anchor_feat_t, ode_flow_t, selector)
    assert solved.shape == (2, 3, 4, 8, 8)
    assert torch.allclose(dynamic_anchor_t, anchor_feat_t, atol=1.0e-6)


def test_cardia_forward_contract_and_diagnostics():
    model = CARDIA(_cfg())
    data = {
        "rgb": torch.rand(2, 3, 1, 64, 64),
        "info": {"num_objects": torch.ones(2, dtype=torch.long)},
    }
    out = model(data)
    assert out["masks_0"].shape == (2, 1, 64, 64)
    assert out["logits_2"].shape == (2, 2, 64, 64)
    aux = out["memory_aux_1"]["cardia_aux"]
    assert aux["proposal_logits"].shape == (2, 1, 3, 64, 64)
    assert aux["head_weights"].shape == (2, 1, 3)
    assert aux["selector_logits"].shape == (2, 1, 3)
    assert aux["stage3_gamma"].item() >= 0.0
    assert aux["stage2_gamma"].item() >= 0.0
    assert aux["boundary_gamma"].item() >= 0.0
    assert aux["runtime_state_detached"].item() == 1.0
    assert aux["stage3_decay_mean"].shape == (2,)
    assert aux["stage2_head_usage"].shape == (2, 3)
    assert aux["boundary_edge_gate"].shape == (2, 1, 64, 64)
    assert aux["boundary_edge_gate_mean"].shape == (2,)


def test_registry_builds_cardia_aliases():
    cfg = OmegaConf.create({"model": _cfg()})
    for alias in ("cardia", "unext_cardia"):
        cfg.model.name = alias
        model = MODEL_REGISTRY.build(cfg, device=torch.device("cpu"))
        assert isinstance(model, CARDIA)
