from __future__ import annotations

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from model.cardia import CARDIA
from model.modules.unext import UNeXtOfficialBackbone


def _official_cardia_cfg():
    return OmegaConf.create(
        {
            "name": "cardia",
            "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
            "temporal_memory": {"type": "none", "bpm": {"ENABLE": False}},
            "cardia": {
                "backbone": {
                    "name": "official",
                    "official": {
                        "mlp_expansion": 2.0,
                        "latent_blocks": 1,
                        "decoder_mlp_blocks": 1,
                    },
                },
                "in_channels": 1,
                "num_classes": 2,
                "base_dim": 8,
                "value_dim": 16,
                "hidden_dim": 8,
                "runtime_token_dim": 8,
                "stage3_num_heads": 1,
                "stage2_num_heads": 3,
                "stage3_max_offset_px": 1.5,
                "stage2_max_offset_px": 3.0,
                "padding_mode": "border",
                "align_corners": False,
                "detach_runtime_state": True,
                "stage3_decay_gate": True,
                "stage2_head_scales": [0.5, 1.0, 1.5],
                "lambda_cardia_base": 0.1,
                "lambda_cardia_proposal_oracle": 0.2,
            },
        }
    )


def test_official_unext_forward_shape_and_cardia_feature_contract():
    model = UNeXtOfficialBackbone(in_channels=1, num_classes=2, base_dim=8, value_dim=16, mlp_expansion=2.0)
    out = model(torch.randn(2, 1, 64, 64))
    assert out["logits"].shape == (2, 2, 64, 64)
    assert out["low"].shape == (2, 8, 32, 32)
    assert out["mid"].shape == (2, 16, 16, 16)
    assert out["high"].shape == (2, 32, 8, 8)
    dec_mid = model.up1(out["high"], out["mid"])
    assert dec_mid.shape == out["mid"].shape


def test_official_unext_handles_non_256_input():
    model = UNeXtOfficialBackbone(in_channels=1, num_classes=2, base_dim=8, value_dim=16, mlp_expansion=2.0)
    out = model(torch.randn(1, 1, 72, 80))
    assert out["logits"].shape == (1, 2, 72, 80)
    assert out["low"].shape[-2:] == (36, 40)


def test_official_unext_base_loss_backward():
    model = UNeXtOfficialBackbone(in_channels=1, num_classes=2, base_dim=8, value_dim=16, mlp_expansion=2.0)
    logits = model(torch.randn(2, 1, 64, 64))["logits"]
    target = torch.randint(0, 2, (2, 64, 64), dtype=torch.long)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    assert model.logit_head.weight.grad is not None
    assert float(model.logit_head.weight.grad.abs().sum().item()) > 0.0


def test_cardia_with_official_unext_forward_and_gradient():
    model = CARDIA(_official_cardia_cfg())
    data = {
        "rgb": torch.rand(1, 2, 1, 64, 64),
        "info": {"num_objects": torch.ones(1, dtype=torch.long)},
    }
    out = model(data)
    assert out["logits_1"].shape == (1, 2, 64, 64)
    aux = out["memory_aux_1"]["cardia_aux"]
    assert aux["proposal_logits"].shape == (1, 1, 3, 64, 64)
    loss = out["logits_1"][:, 1].mean()
    loss.backward()
    assert model.ode_gen2.offset_head.weight.grad is not None
    assert float(model.ode_gen2.offset_head.weight.grad.abs().sum().item()) > 0.0


def test_official_unext_parameter_count_scales_reasonably():
    model = UNeXtOfficialBackbone(in_channels=1, num_classes=2, base_dim=32, value_dim=64)
    params_m = sum(p.numel() for p in model.parameters()) / 1.0e6
    assert 0.5 < params_m < 8.0


def test_official_unext_partial_checkpoint_load_into_cardia(tmp_path):
    src = CARDIA(_official_cardia_cfg())
    ckpt_path = tmp_path / "official_anchor.pt"
    torch.save({"model": {f"backbone.{k}": v.detach().clone() for k, v in src.backbone.state_dict().items()}}, ckpt_path)
    cfg = _official_cardia_cfg()
    cfg.cardia.pretrained_unext_path = str(ckpt_path)
    cfg.cardia.pretrained_unext_strict_backbone = True
    dst = CARDIA(cfg)
    first_key = next(iter(src.backbone.state_dict().keys()))
    assert torch.allclose(src.backbone.state_dict()[first_key], dst.backbone.state_dict()[first_key])
