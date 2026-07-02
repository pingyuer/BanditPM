import torch
from omegaconf import OmegaConf

from debel.grid import grid_sample_logits
from losses.computer import LossComputer
from models.registry import build_model
from utils.frame_validity import build_single_target_mask, summarize_frame_mask


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "debel",
                "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
                "debel": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "d_model": 24,
                    "temporal_layers": 1,
                    "temporal_heads": 4,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "spatial_token_hw": 4,
                    "summary_tokens": 2,
                    "solver_queries": 3,
                    "solver_steps": 2,
                    "max_disp": 0.05,
                    "grid_head_channels": 16,
                    "align_corners": True,
                    "padding_mode": "border",
                    "use_residual": True,
                    "residual_alpha_max": 0.2,
                    "backbone": {
                        "name": "official",
                        "base_dim": 8,
                        "mlp_expansion": 2.0,
                        "latent_blocks": 1,
                        "decoder_mlp_blocks": 1,
                    },
                },
            },
            "loss": {
                "name": "debel",
                "debel": {
                    "lambda_final": 1.0,
                    "lambda_anchor": 0.5,
                    "lambda_grid": 0.01,
                    "lambda_smooth": 0.02,
                    "lambda_temp": 0.01,
                    "lambda_area": 0.001,
                    "lambda_residual": 0.005,
                },
            },
        }
    )


def test_debel_forward_shape_and_bounded_grid():
    model = build_model(_cfg(), device="cpu")
    out = model({"rgb": torch.randn(2, 3, 1, 32, 32)})
    assert out["logits"].shape == (2, 3, 2, 32, 32)
    assert out["anchor_logits"].shape == (2, 3, 2, 32, 32)
    assert out["warped_logits"].shape == (2, 3, 2, 32, 32)
    assert out["delta_grids"].shape == (2, 3, 2, 2, 32, 32)
    assert out["delta_grids"].abs().max().item() <= 0.1001
    assert "debel/query/attn_entropy" in out["aux"]


def test_debel_initial_grid_solver_is_identity_before_training():
    model = build_model(_cfg(), device="cpu")
    out = model({"rgb": torch.randn(1, 2, 1, 32, 32)})
    assert torch.allclose(out["anchor_logits"], out["warped_logits"], atol=1e-5)


def test_debel_grid_sample_identity_helper():
    logits = torch.randn(2, 2, 16, 16)
    delta = torch.zeros(2, 2, 16, 16)
    warped = grid_sample_logits(logits, delta, align_corners=True)
    assert torch.allclose(logits, warped, atol=1e-5)


def test_debel_loss_and_gradients_reach_query_and_grid():
    cfg = _cfg()
    stage = OmegaConf.create({"point_supervision": False, "train_num_points": 16, "oversample_ratio": 3, "importance_sample_ratio": 0.75})
    model = build_model(cfg, device="cpu")
    data = {
        "rgb": torch.randn(1, 2, 1, 32, 32),
        "cls_gt": torch.randint(0, 2, (1, 2, 32, 32)),
    }
    out = model(data)
    data.update(out)
    data["supervised_indices"] = torch.ones(1, 2, dtype=torch.bool)
    losses = LossComputer(cfg, stage).compute(data, [1])
    for key in ("debel_final", "debel_anchor", "debel_grid", "debel_smooth", "debel_temp", "debel_area", "debel_residual"):
        assert key in losses
    losses["total_loss"].backward()
    assert model.query_decoder.queries.grad is not None
    assert model.grid_solver.head[-1].weight.grad is not None


def test_causal_target10_mask_selects_only_last_frame():
    mask = build_single_target_mask(batch_size=2, total_frames=10, target_index=-1)
    assert summarize_frame_mask(mask) == [[9], [9]]
    assert mask.sum().item() == 2

    explicit = build_single_target_mask(batch_size=1, total_frames=10, target_index=7)
    assert summarize_frame_mask(explicit) == [[7]]
