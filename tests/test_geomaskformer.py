from __future__ import annotations

import torch
from omegaconf import OmegaConf

from geomaskformer import DualStreamFactorizedTransformer, GeoMaskFormer, ImageTokenizer, MaskTokenizer, ProposalDecoder
from losses import LossComputer
from losses.geomaskformer import boundary_geometry_loss, compute_geomaskformer_losses
from models.registry import build_model


def _model_cfg(dim: int = 32, queries: int = 6):
    return OmegaConf.create(
        {
            "model": {
                "name": "geomaskformer",
                "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
                "temporal_memory": {"type": "none", "bpm": {"ENABLE": False}},
                "geomaskformer": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "dim": dim,
                    "base_channels": 8,
                    "depth": 1,
                    "heads": 4,
                    "num_queries": queries,
                    "decoder_layers": 1,
                    "max_frames": 8,
                    "training_stage": "stage2",
                    "image_token_dropout": 0.0,
                    "mask_token_dropout": 0.0,
                    "condition_dropout": 0.0,
                    "loss": {"mask": 1.0, "boundary": 0.2, "score": 0.5, "temporal": 0.01, "topk": 2},
                },
            }
        }
    )


def _stage_cfg():
    return OmegaConf.create(
        {
            "point_supervision": False,
            "train_num_points": 32,
            "oversample_ratio": 3.0,
            "importance_sample_ratio": 0.75,
        }
    )


def _batch(batch_size: int = 2, frames: int = 3, size: int = 32):
    rgb = torch.randn(batch_size, frames, 1, size, size)
    cls_gt = torch.zeros(batch_size, frames, 1, size, size, dtype=torch.long)
    cls_gt[:, :, :, 8:24, 10:22] = 1
    label_valid = torch.zeros(batch_size, frames, dtype=torch.bool)
    label_valid[:, 1:] = True
    return {
        "rgb": rgb,
        "cls_gt": cls_gt,
        "label_valid": label_valid,
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
        "geomaskformer_mask_visibility": torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.long),
        "geomaskformer_loss_visibility": label_valid,
    }


def test_mask_tokenizer_geometry_range_and_mask_token_grad():
    tokenizer = MaskTokenizer(dim=32, max_frames=8, use_geometry=True)
    masks = torch.zeros(2, 3, 1, 32, 32)
    masks[:, 0, :, 8:24, 8:24] = 1.0
    visibility = torch.tensor([[1, 0, 0], [0, 1, 0]])
    tokens, aux = tokenizer(masks, visibility, (4, 4))
    assert tokens.shape == (2, 3, 16, 32)
    geom = aux["geometry"]
    assert geom.shape[1] == 3
    assert geom[:, 1].min().item() >= -1.0
    assert geom[:, 1].max().item() <= 1.0
    invisible = tokens[0, 1].mean()
    invisible.backward()
    assert tokenizer.mask_token.grad is not None
    assert tokenizer.mask_token.grad.abs().sum().item() > 0.0


def test_dual_stream_gate_initially_protects_image_stream_then_allows_gradient():
    transformer = DualStreamFactorizedTransformer(dim=32, depth=1, heads=4)
    x = torch.randn(2, 3, 5, 32, requires_grad=True)
    m = torch.randn(2, 3, 5, 32, requires_grad=True)
    x1, m1, aux = transformer(x, m)
    assert (m1 - m).abs().mean().item() > 0.0
    assert aux["block0_image_from_mask_gate"].item() < 1.0e-4
    transformer.blocks[0].raw_x_from_m_gate.data.fill_(4.0)
    x2, _, _ = transformer(x.detach().clone().requires_grad_(True), m.detach().clone().requires_grad_(True))
    loss = x2.mean()
    loss.backward()
    assert transformer.blocks[0].raw_x_from_m_gate.grad is not None


def test_proposal_decoder_quality_bias_and_backprop():
    decoder = ProposalDecoder(dim=32, pixel_dim=32, num_queries=6, num_layers=1, heads=4, max_frames=8)
    x = torch.randn(2, 3, 16, 32)
    m = torch.randn(2, 3, 16, 32)
    pixels = torch.randn(2, 3, 32, 16, 16)
    out = decoder(x, m, pixels)
    assert out["proposal_logits"].shape == (2, 3, 6, 16, 16)
    assert torch.sigmoid(out["quality_scores"]).mean().item() < 0.25
    loss = out["proposal_logits"].mean() + out["quality_scores"].mean()
    loss.backward()
    assert decoder.query_embed.grad is not None
    assert decoder.query_embed.grad.abs().sum().item() > 0.0


def test_boundary_loss_penalizes_shifted_mask_more_than_aligned_mask():
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0
    aligned = torch.full_like(target, -5.0)
    aligned[:, :, 8:24, 8:24] = 5.0
    shifted = torch.full_like(target, -5.0)
    shifted[:, :, 8:24, 12:28] = 5.0
    assert boundary_geometry_loss(aligned, target).item() < boundary_geometry_loss(shifted, target).item()


def test_geomaskformer_forward_loss_backward_and_visibility_decoupling():
    cfg = _model_cfg()
    model = GeoMaskFormer(cfg.model)
    data = _batch()
    out = model(data)
    data.update(out)
    assert out["logits_1"].shape == (2, 2, 32, 32)
    assert out["proposal_logits"].shape == (2, 3, 6, 32, 32)
    assert out["mask_visibility"][0, 0].item() == 1
    assert out["loss_visibility"][0, 0].item() == 0

    lc = LossComputer(cfg, _stage_cfg())
    losses = lc.compute(data, [1, 1])
    assert "aux_geomaskformer_bestofk_mask" in losses
    losses["total_loss"].backward()
    assert model.mask_tokenizer.mask_token.grad is not None
    assert model.proposal_decoder.query_embed.grad is not None
    assert model.image_tokenizer.patch_proj.weight.grad is not None


def test_registry_builds_geomaskformer():
    model = build_model(_model_cfg(), device="cpu")
    assert isinstance(model, GeoMaskFormer)


def test_bestofk_selects_oracle_proposal_independent_of_quality_score():
    cfg = _model_cfg(queries=3)
    cfg.model.geomaskformer.loss.topk = 1
    lc = LossComputer(cfg, _stage_cfg())
    gt = torch.zeros(1, 1, 1, 16, 16, dtype=torch.long)
    gt[:, :, :, 4:12, 4:12] = 1
    proposals = torch.full((1, 1, 3, 16, 16), -5.0)
    proposals[:, :, 0, 0:8, 0:8] = 5.0
    proposals[:, :, 1, 4:12, 4:12] = 5.0
    proposals[:, :, 2, 8:16, 8:16] = 5.0
    data = {
        "rgb": torch.randn(1, 1, 1, 16, 16),
        "cls_gt": gt,
        "proposal_logits": proposals.requires_grad_(True),
        "quality_scores": torch.tensor([[[5.0, -5.0, -5.0]]], requires_grad=True),
    }
    supervised = torch.ones(1, 1, dtype=torch.bool)
    losses = compute_geomaskformer_losses(lc, data, supervised)
    assert losses["geomaskformer/proposal_oracle_topk_dice"] > losses["geomaskformer/proposal_top1_dice"]
    losses["aux_geomaskformer_bestofk_mask"].backward(retain_graph=True)
    grad = data["proposal_logits"].grad.abs().sum(dim=(0, 1, 3, 4))
    assert grad[1].item() > grad[0].item()
