from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from geomaskformer import (
    DualStreamFactorizedTransformer,
    DeterministicMaskRefiner,
    FullResolutionProposalRefiner,
    GeoMaskFormer,
    ImageTokenizer,
    MaskTokenizer,
    PromptQueryAdapter,
    ProposalCascadeRefiner,
    ProposalDecoder,
    FullResolutionProposalHead,
)
from losses import LossComputer
from losses.geomaskformer import boundary_geometry_loss, compute_geomaskformer_losses
from models.registry import build_model
from training.parameter_groups import get_parameter_groups


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
                    "mask_prompt_noise_prob": 0.0,
                    "mask_prompt_block_prob": 0.0,
                    "mask_prompt_block_ratio": 0.0,
                    "loss": {
                        "mask": 1.0,
                        "boundary": 0.2,
                        "score": 0.5,
                        "temporal": 0.01,
                        "diversity": 0.01,
                        "visible_reconstruction": 0.0,
                        "topk": 2,
                    },
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
    visibility = torch.zeros(batch_size, frames, dtype=torch.long)
    if batch_size > 0:
        visibility[0, 0] = 1
    if batch_size > 1:
        visibility[1, min(1, frames - 1)] = 1
    return {
        "rgb": rgb,
        "cls_gt": cls_gt,
        "label_valid": label_valid,
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
        "geomaskformer_mask_visibility": visibility,
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


def test_mask_tokenizer_invisible_masks_are_content_invariant_and_visible_masks_matter():
    tokenizer = MaskTokenizer(dim=32, max_frames=8, use_geometry=True)
    visibility = torch.tensor([[0, 1]], dtype=torch.long)
    zeros = torch.zeros(1, 2, 1, 32, 32)
    ones = zeros.clone()
    ones[:, 0] = 1.0
    noise = zeros.clone()
    noise[:, 0] = torch.rand_like(noise[:, 0])
    tokens_zero, _ = tokenizer(zeros, visibility, (4, 4))
    tokens_one, _ = tokenizer(ones, visibility, (4, 4))
    tokens_noise, _ = tokenizer(noise, visibility, (4, 4))
    assert torch.equal(tokens_zero[:, 0], tokens_one[:, 0])
    assert torch.equal(tokens_zero[:, 0], tokens_noise[:, 0])

    visible_a = zeros.clone()
    visible_b = zeros.clone()
    visible_b[:, 1, :, 8:24, 8:24] = 1.0
    tokens_a, _ = tokenizer(visible_a, visibility, (4, 4))
    tokens_b, _ = tokenizer(visible_b, visibility, (4, 4))
    assert not torch.allclose(tokens_a[:, 1], tokens_b[:, 1])


def test_mask_tokenizer_geometry_channels_are_local_prior_not_sdf():
    tokenizer = MaskTokenizer(dim=32, max_frames=8, use_geometry=True)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.0
    geom = tokenizer.geometry(mask)
    assert geom.shape[1] == 3
    assert geom[:, 1].min().item() >= -1.0
    assert geom[:, 1].max().item() <= 1.0
    assert geom[:, 1, 16, 16].item() > geom[:, 1, 0, 0].item()
    assert geom[:, 2].max().item() > 0.0


def test_mask_tokenizer_boundary_band_is_localized_on_contour():
    tokenizer = MaskTokenizer(dim=32, max_frames=8, use_geometry=True)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.0
    boundary = tokenizer.geometry(mask)[:, 2]
    assert boundary[:, 8, 16].item() > 0.5
    assert boundary[:, 16, 16].item() == 0.0
    assert boundary[:, 2, 2].item() == 0.0


def test_image_tokenizer_groupnorm_batch_size_one_backward_is_stable():
    tokenizer = ImageTokenizer(in_channels=1, dim=32, base_channels=8)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in tokenizer.modules())
    images = torch.randn(1, 2, 1, 32, 32, requires_grad=True)
    out = tokenizer(images)
    loss = out["tokens"].mean() + out["f2"].mean()
    loss.backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


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


def test_prompt_query_adapter_prompt_no_prompt_and_content_paths_backprop():
    adapter = PromptQueryAdapter(dim=32, num_queries=6, max_frames=8)
    x = torch.randn(2, 3, 16, 32, requires_grad=True)
    m = torch.randn(2, 3, 16, 32, requires_grad=True)
    no_visibility = torch.zeros(2, 3, dtype=torch.long)
    first_visibility = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.long)
    near = torch.zeros(2, 3, dtype=torch.long)
    no_prompt, no_context, no_aux = adapter(x, m, no_visibility, near, near)
    prompted, prompt_context, aux = adapter(x, m, first_visibility, near, near)
    assert no_prompt.shape == (2, 3, 6, 32)
    assert prompt_context.shape == (2, 3, 7, 32)
    assert aux["has_visible_prompt_ratio"].item() == 1.0
    assert no_aux["no_prompt_used_ratio"].item() == 1.0
    assert not torch.allclose(no_prompt, prompted)
    assert not torch.allclose(no_context, prompt_context)
    m_changed = m.detach().clone()
    m_changed[:, 0, :, 0] = m_changed[:, 0, :, 0] + 1.0
    prompted_changed, _, _ = adapter(x.detach(), m_changed.requires_grad_(True), first_visibility, near, near)
    assert not torch.allclose(prompted, prompted_changed)
    (no_prompt.mean() + prompted.mean()).backward()
    assert adapter.no_prompt.grad is not None
    assert adapter.prompt_to_queries[-1].weight.grad is not None
    assert m.grad is not None


def test_prompt_query_adapter_ignores_invisible_token_content():
    adapter = PromptQueryAdapter(dim=32, num_queries=6, max_frames=8)
    x = torch.randn(1, 3, 16, 32)
    m = torch.randn(1, 3, 16, 32)
    visibility = torch.tensor([[1, 0, 0]], dtype=torch.long)
    near = torch.zeros(1, 3, dtype=torch.long)
    base, _, _ = adapter(x, m, visibility, near, near)
    changed = m.clone()
    changed[:, 1:] = changed[:, 1:] + torch.randn_like(changed[:, 1:]) * 20.0
    out, _, _ = adapter(x, changed, visibility, near, near)
    assert torch.allclose(base, out)


def test_proposal_decoder_quality_bias_and_backprop():
    decoder = ProposalDecoder(dim=32, pixel_dim=32, num_queries=6, num_layers=1, heads=4, max_frames=8)
    assert decoder.mask_embed[-1].weight.std().item() < 0.05
    assert decoder.quality[-1].weight.std().item() < 0.05
    assert torch.sigmoid(decoder.quality[-1].bias).item() < 0.15
    x = torch.randn(2, 3, 16, 32)
    m = torch.randn(2, 3, 16, 32)
    pixels = torch.randn(2, 3, 32, 16, 16)
    prompt_bias = torch.randn(2, 3, 6, 32, requires_grad=True)
    out = decoder(x, m, pixels, prompt_query_bias=prompt_bias)
    assert out["proposal_logits"].shape == (2, 3, 6, 16, 16)
    assert torch.sigmoid(out["quality_scores"]).mean().item() < 0.25
    loss = out["proposal_logits"].mean() + out["quality_scores"].mean()
    loss.backward()
    assert decoder.query_embed.grad is not None
    assert decoder.query_embed.grad.abs().sum().item() > 0.0
    assert prompt_bias.grad is not None


def test_proposal_decoder_prompt_bias_changes_mask_proposals():
    decoder = ProposalDecoder(dim=32, pixel_dim=32, num_queries=6, num_layers=1, heads=4, max_frames=8)
    x = torch.randn(1, 3, 16, 32)
    m = torch.randn(1, 3, 16, 32)
    pixels = torch.randn(1, 3, 32, 16, 16)
    zero_prompt = torch.zeros(1, 3, 6, 32)
    active_prompt = torch.randn(1, 3, 6, 32)
    out_zero = decoder(x, m, pixels, prompt_query_bias=zero_prompt)
    out_active = decoder(x, m, pixels, prompt_query_bias=active_prompt)
    assert (out_zero["proposal_logits"] - out_active["proposal_logits"]).abs().mean().item() > 1.0e-5


def test_proposal_decoder_prompt_context_gate_changes_mask_proposals_and_backprops():
    decoder = ProposalDecoder(dim=32, pixel_dim=32, num_queries=6, num_layers=1, heads=4, max_frames=8)
    x = torch.randn(1, 3, 16, 32)
    m = torch.randn(1, 3, 16, 32)
    pixels = torch.randn(1, 3, 32, 16, 16)
    prompt_bias = torch.zeros(1, 3, 6, 32, requires_grad=True)
    prompt_context_a = torch.zeros(1, 3, 7, 32, requires_grad=True)
    prompt_context_b = torch.randn(1, 3, 7, 32)
    out_a = decoder(x, m, pixels, prompt_query_bias=prompt_bias, prompt_context=prompt_context_a, use_prompt_gate=True)
    out_b = decoder(x, m, pixels, prompt_query_bias=prompt_bias.detach(), prompt_context=prompt_context_b, use_prompt_gate=True)
    assert (out_a["proposal_logits"] - out_b["proposal_logits"]).abs().mean().item() > 1.0e-5
    (out_a["proposal_logits"].mean() + out_a["quality_scores"].mean()).backward()
    assert prompt_context_a.grad is not None
    assert decoder.prompt_cross.in_proj_weight.grad is not None


def test_cascade_refiner_changes_logits_and_backprops_to_query_and_pixel():
    refiner = ProposalCascadeRefiner(dim=32, pixel_dim=32)
    logits = torch.randn(1, 2, 4, 8, 8, requires_grad=True)
    pixels = torch.randn(1, 2, 32, 8, 8, requires_grad=True)
    query = torch.randn(1, 2, 4, 32, requires_grad=True)
    refined = refiner(logits, pixels, query)
    assert refined.shape == logits.shape
    loss = boundary_geometry_loss(refined[:, 0, :1].unsqueeze(1).reshape(1, 1, 8, 8), torch.sigmoid(logits[:, 0, :1]).detach().reshape(1, 1, 8, 8))
    loss.backward()
    assert pixels.grad is not None and pixels.grad.abs().sum().item() > 0.0
    assert query.grad is not None and query.grad.abs().sum().item() > 0.0


def test_diffusion_refiner_steps_are_supervisable_and_backpropagate():
    refiner = DeterministicMaskRefiner(dim=32, pixel_dim=32, steps=3)
    logits = torch.randn(1, 2, 4, 8, 8, requires_grad=True)
    pixels = torch.randn(1, 2, 32, 8, 8, requires_grad=True)
    query = torch.randn(1, 2, 4, 32, requires_grad=True)
    final, steps = refiner(logits, pixels, query)
    assert final.shape == logits.shape
    assert len(steps) == 3
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    losses = [boundary_geometry_loss(step[:, 0, :1].reshape(1, 1, 8, 8), target) for step in steps]
    total = torch.stack(losses).mean()
    total.backward()
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0.0
    assert pixels.grad is not None and pixels.grad.abs().sum().item() > 0.0
    assert query.grad is not None and query.grad.abs().sum().item() > 0.0


def test_full_resolution_refiner_outputs_eval_resolution_and_backprops():
    refiner = FullResolutionProposalRefiner(dim=32, f1_channels=8, ref_dim=8)
    lowres = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    query = torch.randn(1, 2, 4, 32, requires_grad=True)
    f1 = torch.randn(2, 8, 16, 16, requires_grad=True)
    images = torch.randn(1, 2, 1, 16, 16)
    fullres = refiner(lowres, query, f1, images)
    assert fullres.shape == (1, 2, 4, 16, 16)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    loss = boundary_geometry_loss(fullres[:, 0, :1].reshape(1, 1, 16, 16), target)
    loss.backward()
    assert lowres.grad is not None and lowres.grad.abs().sum().item() > 0.0
    assert query.grad is not None and query.grad.abs().sum().item() > 0.0
    assert f1.grad is not None and f1.grad.abs().sum().item() > 0.0


def test_full_resolution_proposal_head_generates_query_conditioned_fullres_masks():
    head = FullResolutionProposalHead(dim=32, f1_channels=8, lowres_channels=32, ref_dim=8)
    lowres = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    lowres_pixel = torch.randn(1, 2, 32, 4, 4, requires_grad=True)
    query = torch.randn(1, 2, 4, 32, requires_grad=True)
    f1 = torch.randn(2, 8, 16, 16, requires_grad=True)
    images = torch.randn(1, 2, 1, 16, 16)
    fullres = head(lowres, lowres_pixel, query, f1, images)
    assert fullres.shape == (1, 2, 4, 16, 16)
    changed_query = query.detach().clone()
    changed_query[:, :, 0, 0] = changed_query[:, :, 0, 0] + 5.0
    changed = head(lowres.detach(), lowres_pixel.detach(), changed_query, f1.detach(), images)
    assert (fullres[:, :, 0] - changed[:, :, 0]).abs().mean().item() > 1.0e-5
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    loss = boundary_geometry_loss(fullres[:, 0, :1].reshape(1, 1, 16, 16), target)
    loss.backward()
    assert lowres.grad is not None and lowres.grad.abs().sum().item() > 0.0
    assert lowres_pixel.grad is not None and lowres_pixel.grad.abs().sum().item() > 0.0
    assert query.grad is not None and query.grad.abs().sum().item() > 0.0
    assert f1.grad is not None and f1.grad.abs().sum().item() > 0.0


def test_proposal_decoder_condition_distance_changes_queries():
    decoder = ProposalDecoder(dim=32, pixel_dim=32, num_queries=6, num_layers=1, heads=4, max_frames=8)
    x = torch.randn(1, 3, 16, 32)
    m = torch.randn(1, 3, 16, 32)
    pixels = torch.randn(1, 3, 32, 16, 16)
    near = torch.zeros(1, 3, dtype=torch.long)
    far = torch.full((1, 3), 3, dtype=torch.long)
    out_near = decoder(x, m, pixels, condition_prev=near, condition_next=near)
    out_far = decoder(x, m, pixels, condition_prev=far, condition_next=far)
    assert not torch.allclose(out_near["quality_scores"], out_far["quality_scores"])


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
    assert out["proposal_logits_lowres"].shape[-2:] != out["proposal_logits"].shape[-2:]
    assert out["mask_visibility"][0, 0].item() == 1
    assert out["loss_visibility"][0, 0].item() == 0

    lc = LossComputer(cfg, _stage_cfg())
    losses = lc.compute(data, [1, 1])
    assert "aux_geomaskformer_bestofk_mask" in losses
    losses["total_loss"].backward()
    assert model.mask_tokenizer.mask_token.grad is not None
    assert model.prompt_query_adapter.prompt_to_queries[-1].weight.grad is not None
    assert model.proposal_decoder.query_embed.grad is not None
    assert model.full_res_refiner is not None
    assert model.full_res_refiner.local_refine[-1].weight.grad is not None
    assert model.image_tokenizer.patch_proj.weight.grad is not None


def test_geomaskformer_variants_forward_loss_backward_have_variant_gradients():
    for variant in ("v2_prompt_gate", "cascade_refine", "diffusion_refine", "fullres_proposal", "fullres_cascade"):
        cfg = _model_cfg()
        cfg.model.geomaskformer.architecture_variant = variant
        cfg.model.geomaskformer.loss.refinement = 0.1
        model = GeoMaskFormer(cfg.model)
        data = _batch(batch_size=1)
        out = model(data)
        data.update(out)
        losses = LossComputer(cfg, _stage_cfg()).compute(data, [1])
        losses["total_loss"].backward()
        assert model.prompt_query_adapter.prompt_to_queries[-1].weight.grad is not None
        assert model.proposal_decoder.query_embed.grad is not None
        if variant in {"fullres_proposal", "fullres_cascade"}:
            assert model.full_res_proposal_head is not None
            assert model.full_res_proposal_head.shallow[0].weight.grad is not None
            assert model.full_res_proposal_head.shallow[0].weight.grad.abs().sum().item() > 0.0
        else:
            assert model.full_res_refiner is not None
            assert model.full_res_refiner.shallow[0].weight.grad is not None
        if variant == "v2_prompt_gate":
            assert model.proposal_decoder.prompt_cross.in_proj_weight.grad is not None
            assert model.proposal_decoder.prompt_cross.in_proj_weight.grad.abs().sum().item() > 0.0
        if variant in {"cascade_refine", "diffusion_refine", "fullres_cascade"}:
            assert model.variant_refiner is not None
            variant_grad = sum(
                float(p.grad.abs().sum().item())
                for p in model.variant_refiner.parameters()
                if p.grad is not None
            )
            assert variant_grad > 0.0
        if variant == "diffusion_refine":
            assert "aux_geomaskformer_refinement" in losses


def test_default_masked_completion_loss_excludes_visible_condition_frames():
    cfg = _model_cfg()
    model = GeoMaskFormer(cfg.model)
    model.train()
    data = _batch()
    data.pop("geomaskformer_loss_visibility")
    data["label_valid"][:] = True
    data["geomaskformer_mask_visibility"] = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.long)
    out = model(data)
    assert out["loss_visibility"][0].tolist() == [False, True, True]
    assert out["loss_visibility"][1].tolist() == [True, False, True]


def test_condition_mask_perturbation_changes_masked_prediction():
    cfg = _model_cfg()
    model = GeoMaskFormer(cfg.model)
    model.eval()
    data_a = _batch(batch_size=1)
    data_a["geomaskformer_mask_visibility"] = torch.tensor([[1, 0, 0]], dtype=torch.long)
    data_a["geomaskformer_loss_visibility"] = torch.tensor([[False, True, True]])
    data_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in data_a.items()}
    data_b["cls_gt"] = data_a["cls_gt"].clone()
    data_b["cls_gt"][:, 0] = 0
    data_b["cls_gt"][:, 0, :, 2:16, 2:16] = 1
    with torch.no_grad():
        full_a = model(data_a)
        full_b = model(data_b)
        out_a = full_a["logits"][:, 1:]
        out_b = full_b["logits"][:, 1:]
        proposals_a = full_a["proposal_logits"][:, 1:]
        proposals_b = full_b["proposal_logits"][:, 1:]
    assert (out_a - out_b).abs().mean().item() > 1.0e-6
    assert (proposals_a - proposals_b).abs().mean().item() > 1.0e-6


def test_eval_default_visibility_is_all_mask_invisible_without_explicit_prompt():
    cfg = _model_cfg()
    model = GeoMaskFormer(cfg.model)
    model.eval()
    data = _batch(batch_size=1)
    data.pop("geomaskformer_mask_visibility")
    data.pop("geomaskformer_loss_visibility")
    data["label_valid"][:] = True
    with torch.no_grad():
        out = model(data)
    assert out["mask_visibility"].sum().item() == 0
    assert out["loss_visibility"].all().item()


def test_visible_prompt_mask_corruption_is_train_only_and_does_not_mutate_targets():
    cfg = _model_cfg()
    cfg.model.geomaskformer.mask_prompt_noise_prob = 1.0
    cfg.model.geomaskformer.mask_prompt_block_prob = 1.0
    cfg.model.geomaskformer.mask_prompt_block_ratio = 0.25
    model = GeoMaskFormer(cfg.model)
    data = _batch(batch_size=1)
    data["geomaskformer_mask_visibility"] = torch.tensor([[1, 0, 0]], dtype=torch.long)
    original_target = data["cls_gt"].clone()

    model.train()
    train_out = model(data)
    train_aux = train_out["geomaskformer_aux"]
    assert train_aux["mask_prompt_pixel_corruption_ratio"].item() > 0.95
    assert train_aux["mask_prompt_block_corruption_ratio"].item() == 1.0
    assert torch.equal(data["cls_gt"], original_target)
    assert train_out["loss_visibility"][0].tolist() == [False, True, True]

    model.eval()
    with torch.no_grad():
        eval_out = model(data)
    eval_aux = eval_out["geomaskformer_aux"]
    assert eval_aux["mask_prompt_pixel_corruption_ratio"].item() == 0.0
    assert eval_aux["mask_prompt_block_corruption_ratio"].item() == 0.0
    assert torch.equal(data["cls_gt"], original_target)


def test_prompt_corruption_metrics_are_available_to_training_loss_logging():
    cfg = _model_cfg()
    cfg.model.geomaskformer.mask_prompt_noise_prob = 1.0
    cfg.model.geomaskformer.mask_prompt_block_prob = 1.0
    cfg.model.geomaskformer.mask_prompt_block_ratio = 0.25
    model = GeoMaskFormer(cfg.model)
    model.train()
    data = _batch(batch_size=1)
    data["geomaskformer_mask_visibility"] = torch.tensor([[1, 0, 0]], dtype=torch.long)
    out = model(data)
    data.update(out)
    losses = LossComputer(cfg, _stage_cfg()).compute(data, [1])
    assert losses["geomaskformer/mask_prompt_pixel_corruption_ratio"].item() > 0.95
    assert losses["geomaskformer/mask_prompt_block_corruption_ratio"].item() == 1.0


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
    assert losses["geomaskformer/proposal_oracle_topk_mean_dice"] > losses["geomaskformer/proposal_selected_dice"]
    losses["aux_geomaskformer_bestofk_mask"].backward(retain_graph=True)
    grad = data["proposal_logits"].grad.abs().sum(dim=(0, 1, 3, 4))
    assert grad[1].item() > grad[0].item()


def test_proposal_cover_metrics_use_true_topk_and_top5_values():
    cfg = _model_cfg(queries=6)
    cfg.model.geomaskformer.loss.topk = 3
    lc = LossComputer(cfg, _stage_cfg())
    gt = torch.zeros(1, 1, 1, 16, 16, dtype=torch.long)
    gt[:, :, :, 4:12, 4:12] = 1
    proposals = torch.full((1, 1, 6, 16, 16), -5.0)
    proposals[:, :, 0, 0:8, 0:8] = 5.0
    proposals[:, :, 1, 4:12, 4:12] = 5.0
    proposals[:, :, 2, 6:14, 4:12] = 5.0
    proposals[:, :, 3, 4:12, 6:14] = 5.0
    proposals[:, :, 4, 2:10, 2:10] = 5.0
    proposals[:, :, 5, 12:16, 12:16] = 5.0
    data = {
        "rgb": torch.randn(1, 1, 1, 16, 16),
        "cls_gt": gt,
        "proposal_logits": proposals.requires_grad_(True),
        "quality_scores": torch.zeros(1, 1, 6, requires_grad=True),
    }
    losses = compute_geomaskformer_losses(lc, data, torch.ones(1, 1, dtype=torch.bool))
    assert losses["geomaskformer/proposal_oracle_best_dice"] > losses["geomaskformer/proposal_oracle_topk_mean_dice"]
    assert losses["geomaskformer/proposal_oracle_top5_best_dice"] == losses["geomaskformer/proposal_oracle_best_dice"]
    assert losses["geomaskformer/proposal_oracle_top5_mean_dice"] < losses["geomaskformer/proposal_oracle_top5_best_dice"]
    assert losses["geomaskformer/proposal_top5_cover_rate_0p85"].item() == 1.0
    assert losses["geomaskformer/proposal_top5_cover_rate_0p90"].item() == 1.0
    assert "aux_geomaskformer_diversity" in losses


def test_geomaskformer_score_and_diversity_losses_warm_up():
    cfg = _model_cfg(queries=3)
    cfg.model.geomaskformer.loss.topk = 2
    cfg.model.geomaskformer.loss.score_warmup_iters = 10
    cfg.model.geomaskformer.loss.ranking_warmup_iters = 10
    cfg.model.geomaskformer.loss.diversity_warmup_iters = 10
    lc = LossComputer(cfg, _stage_cfg())
    gt = torch.zeros(1, 1, 1, 16, 16, dtype=torch.long)
    gt[:, :, :, 4:12, 4:12] = 1
    proposals = torch.full((1, 1, 3, 16, 16), -5.0)
    proposals[:, :, 0, 4:12, 4:12] = 5.0
    proposals[:, :, 1, 4:12, 6:14] = 5.0
    proposals[:, :, 2, 0:4, 0:4] = 5.0
    base = {
        "rgb": torch.randn(1, 1, 1, 16, 16),
        "cls_gt": gt,
        "proposal_logits": proposals,
        "quality_scores": torch.zeros(1, 1, 3),
    }
    early = compute_geomaskformer_losses(lc, {**base, "global_step": 0}, torch.ones(1, 1, dtype=torch.bool))
    late = compute_geomaskformer_losses(lc, {**base, "global_step": 9}, torch.ones(1, 1, dtype=torch.bool))
    assert abs(early["geomaskformer/score_warmup_scale"].item() - 0.1) < 1.0e-6
    assert abs(early["geomaskformer/ranking_warmup_scale"].item() - 0.1) < 1.0e-6
    assert abs(late["geomaskformer/score_warmup_scale"].item() - 1.0) < 1.0e-6
    assert early["aux_geomaskformer_score"] < late["aux_geomaskformer_score"]
    assert early["aux_geomaskformer_ranking"] < late["aux_geomaskformer_ranking"]
    assert early["aux_geomaskformer_diversity"] < late["aux_geomaskformer_diversity"]


def test_diversity_loss_penalizes_duplicate_topk_proposals_more_than_varied_ones():
    cfg = _model_cfg(queries=3)
    cfg.model.geomaskformer.loss.topk = 2
    lc = LossComputer(cfg, _stage_cfg())
    gt = torch.zeros(1, 1, 1, 16, 16, dtype=torch.long)
    gt[:, :, :, 4:12, 4:12] = 1

    duplicate = torch.full((1, 1, 3, 16, 16), -5.0)
    duplicate[:, :, 0, 4:12, 4:12] = 5.0
    duplicate[:, :, 1, 4:12, 4:12] = 5.0
    duplicate[:, :, 2, 0:4, 0:4] = 5.0

    varied = duplicate.clone()
    varied[:, :, 1] = -5.0
    varied[:, :, 1, 4:12, 6:14] = 5.0

    common = {
        "rgb": torch.randn(1, 1, 1, 16, 16),
        "cls_gt": gt,
        "quality_scores": torch.zeros(1, 1, 3, requires_grad=True),
    }
    duplicate_losses = compute_geomaskformer_losses(
        lc, {**common, "proposal_logits": duplicate.requires_grad_(True)}, torch.ones(1, 1, dtype=torch.bool)
    )
    varied_losses = compute_geomaskformer_losses(
        lc, {**common, "proposal_logits": varied.requires_grad_(True)}, torch.ones(1, 1, dtype=torch.bool)
    )
    assert duplicate_losses["raw_geomaskformer_diversity"] > varied_losses["raw_geomaskformer_diversity"]


def test_quality_loss_target_matches_detached_proposal_dice():
    cfg = _model_cfg(queries=2)
    cfg.model.geomaskformer.loss.topk = 1
    lc = LossComputer(cfg, _stage_cfg())
    gt = torch.zeros(1, 1, 1, 8, 8, dtype=torch.long)
    gt[:, :, :, 2:6, 2:6] = 1
    proposals = torch.full((1, 1, 2, 8, 8), -4.0)
    proposals[:, :, 0, 2:6, 2:6] = 4.0
    proposals[:, :, 1, 0:4, 0:4] = 4.0
    quality_scores = torch.zeros(1, 1, 2, requires_grad=True)
    data = {
        "rgb": torch.randn(1, 1, 1, 8, 8),
        "cls_gt": gt,
        "proposal_logits": proposals,
        "quality_scores": quality_scores,
    }
    losses = compute_geomaskformer_losses(lc, data, torch.ones(1, 1, dtype=torch.bool))
    target = losses["geomaskformer/proposal_oracle_best_dice"]
    assert target.item() > losses["geomaskformer/proposal_selected_dice"].item() - 1.0
    losses["aux_geomaskformer_score"].backward()
    assert quality_scores.grad is not None


def test_quality_loss_gradient_pushes_better_proposal_score_up():
    cfg = _model_cfg(queries=2)
    cfg.model.geomaskformer.loss.topk = 1
    lc = LossComputer(cfg, _stage_cfg())
    gt = torch.zeros(1, 1, 1, 8, 8, dtype=torch.long)
    gt[:, :, :, 2:6, 2:6] = 1
    proposals = torch.full((1, 1, 2, 8, 8), -4.0)
    proposals[:, :, 0, 2:6, 2:6] = 4.0
    proposals[:, :, 1, 0:4, 0:4] = 4.0
    quality_scores = torch.zeros(1, 1, 2, requires_grad=True)
    data = {
        "rgb": torch.randn(1, 1, 1, 8, 8),
        "cls_gt": gt,
        "proposal_logits": proposals,
        "quality_scores": quality_scores,
    }
    losses = compute_geomaskformer_losses(lc, data, torch.ones(1, 1, dtype=torch.bool))
    losses["aux_geomaskformer_score"].backward()
    assert quality_scores.grad[0, 0, 0].item() < 0.0
    assert quality_scores.grad[0, 0, 1].item() > quality_scores.grad[0, 0, 0].item()


def test_quality_ranking_loss_pushes_known_better_proposal_above_worse_one():
    cfg = _model_cfg(queries=2)
    cfg.model.geomaskformer.loss.topk = 1
    cfg.model.geomaskformer.loss.score = 0.0
    cfg.model.geomaskformer.loss.ranking = 1.0
    lc = LossComputer(cfg, _stage_cfg())
    gt = torch.zeros(1, 1, 1, 8, 8, dtype=torch.long)
    gt[:, :, :, 2:6, 2:6] = 1
    proposals = torch.full((1, 1, 2, 8, 8), -4.0)
    proposals[:, :, 0, 2:6, 2:6] = 4.0
    proposals[:, :, 1, 0:4, 0:4] = 4.0
    quality_scores = torch.zeros(1, 1, 2, requires_grad=True)
    data = {
        "rgb": torch.randn(1, 1, 1, 8, 8),
        "cls_gt": gt,
        "proposal_logits": proposals,
        "quality_scores": quality_scores,
    }
    losses = compute_geomaskformer_losses(lc, data, torch.ones(1, 1, dtype=torch.bool))
    losses["aux_geomaskformer_ranking"].backward()
    assert quality_scores.grad[0, 0, 0].item() < 0.0
    assert quality_scores.grad[0, 0, 1].item() > 0.0


def test_geomaskformer_parameter_groups_specialize_prompt_proposal_and_quality_lr():
    cfg = _model_cfg()
    model = GeoMaskFormer(cfg.model)
    stage_cfg = OmegaConf.create(
        {
            "weight_decay": 0.001,
            "embed_weight_decay": 0.0,
            "backbone_lr_ratio": 0.5,
            "learning_rate": 1.0e-4,
            "geomaskformer_lr_ratio": 1.0,
            "geomaskformer_prompt_lr_mult": 1.5,
            "geomaskformer_proposal_lr_mult": 1.2,
            "geomaskformer_quality_lr_mult": 2.0,
        }
    )
    groups = {group["name"]: group for group in get_parameter_groups(model, stage_cfg)}
    assert abs(groups["geomaskformer_prompt_query_adapter"]["lr"] - 1.5e-4) < 1.0e-12
    assert abs(groups["geomaskformer_proposal_decoder"]["lr"] - 1.2e-4) < 1.0e-12
    assert abs(groups["geomaskformer_quality_head"]["lr"] - 2.0e-4) < 1.0e-12
    assert groups["geomaskformer_quality_head_no_decay"]["weight_decay"] == 0.0
    assert len(groups["geomaskformer_prompt_query_adapter"]["params"]) > 0
    assert len(groups["geomaskformer_proposal_decoder"]["params"]) > 0
    assert len(groups["geomaskformer_quality_head"]["params"]) > 0

    cfg.model.geomaskformer.architecture_variant = "cascade_refine"
    variant_model = GeoMaskFormer(cfg.model)
    variant_groups = {group["name"]: group for group in get_parameter_groups(variant_model, stage_cfg)}
    first_refiner_param = next(variant_model.variant_refiner.parameters())
    proposal_params = (
        variant_groups["geomaskformer_proposal_decoder"]["params"]
        + variant_groups["geomaskformer_proposal_decoder_no_decay"]["params"]
    )
    assert any(p is first_refiner_param for p in proposal_params)
    first_full_res_param = next(variant_model.full_res_refiner.parameters())
    assert any(p is first_full_res_param for p in proposal_params)

    cfg.model.geomaskformer.architecture_variant = "fullres_proposal"
    fullres_model = GeoMaskFormer(cfg.model)
    fullres_groups = {group["name"]: group for group in get_parameter_groups(fullres_model, stage_cfg)}
    fullres_proposal_params = (
        fullres_groups["geomaskformer_proposal_decoder"]["params"]
        + fullres_groups["geomaskformer_proposal_decoder_no_decay"]["params"]
    )
    first_fullres_head_param = next(fullres_model.full_res_proposal_head.parameters())
    assert fullres_model.full_res_refiner is None
    assert any(p is first_fullres_head_param for p in fullres_proposal_params)

    cfg.model.geomaskformer.architecture_variant = "fullres_cascade"
    fullres_cascade_model = GeoMaskFormer(cfg.model)
    fullres_cascade_groups = {group["name"]: group for group in get_parameter_groups(fullres_cascade_model, stage_cfg)}
    fullres_cascade_proposal_params = (
        fullres_cascade_groups["geomaskformer_proposal_decoder"]["params"]
        + fullres_cascade_groups["geomaskformer_proposal_decoder_no_decay"]["params"]
    )
    assert fullres_cascade_model.full_res_refiner is None
    assert fullres_cascade_model.full_res_proposal_head is not None
    assert fullres_cascade_model.variant_refiner is not None
    first_fullres_cascade_head_param = next(fullres_cascade_model.full_res_proposal_head.parameters())
    first_fullres_cascade_variant_param = next(fullres_cascade_model.variant_refiner.parameters())
    assert any(first_fullres_cascade_head_param is p for p in fullres_cascade_proposal_params)
    assert any(first_fullres_cascade_variant_param is p for p in fullres_cascade_proposal_params)


def test_geomaskformer_experiment_config_uses_postprocess_and_lr_multipliers():
    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(version_base="1.3.2", config_dir=config_dir):
        echo_cfg = compose(config_name="geomaskformer_echo")
        echo_gdkvm = compose(config_name="gdkvm_echo")
        camus_cfg = compose(config_name="geomaskformer_camus")
        camus_gdkvm = compose(config_name="gdkvm_camus")
        prompt_gate = compose(config_name="geomaskformer_v2_prompt_gate_echo")
        cascade = compose(config_name="geomaskformer_cascade_refine_echo")
        fullres = compose(config_name="geomaskformer_fullres_proposal_echo")
        fullres_cascade = compose(config_name="geomaskformer_fullres_cascade_echo")
        adult_cascade = compose(config_name="geomaskformer_cascade_refine_echonet_adult")
        pediatric_cascade = compose(config_name="geomaskformer_cascade_refine_echonet_pediatric")
        camus_fullres_cascade = compose(config_name="geomaskformer_fullres_cascade_camus")
        diffusion = compose(config_name="geomaskformer_diffusion_refine_echo")
    for cfg, gdkvm in ((echo_cfg, echo_gdkvm), (camus_cfg, camus_gdkvm)):
        assert cfg.evaluation.protocol_version == gdkvm.evaluation.protocol_version
        assert cfg.evaluation.tta.enabled == gdkvm.evaluation.tta.enabled
        assert list(cfg.evaluation.tta.modes) == list(gdkvm.evaluation.tta.modes)
        assert cfg.evaluation.postprocess.enabled == gdkvm.evaluation.postprocess.enabled
        assert cfg.evaluation.postprocess.largest_component == gdkvm.evaluation.postprocess.largest_component
        assert cfg.evaluation.postprocess.fill_holes == gdkvm.evaluation.postprocess.fill_holes
        assert cfg.evaluation.postprocess.remove_small_objects == gdkvm.evaluation.postprocess.remove_small_objects
        assert cfg.evaluation.postprocess.binary_closing == gdkvm.evaluation.postprocess.binary_closing
        assert int(cfg.evaluation.postprocess.min_size) == int(gdkvm.evaluation.postprocess.min_size)
    cfg = echo_cfg
    assert cfg.evaluation.postprocess.enabled
    assert cfg.evaluation.postprocess.largest_component
    assert cfg.evaluation.postprocess.fill_holes
    assert cfg.evaluation.postprocess.binary_closing
    assert int(cfg.evaluation.postprocess.min_size) == 16
    assert cfg.main_training.geomaskformer_prompt_lr_mult == 1.5
    assert cfg.main_training.geomaskformer_proposal_lr_mult == 1.2
    assert cfg.main_training.geomaskformer_quality_lr_mult == 2.0
    assert prompt_gate.model.geomaskformer.architecture_variant == "v2_prompt_gate"
    assert cascade.model.geomaskformer.architecture_variant == "cascade_refine"
    assert fullres.model.geomaskformer.architecture_variant == "fullres_proposal"
    assert fullres_cascade.model.geomaskformer.architecture_variant == "fullres_cascade"
    assert diffusion.model.geomaskformer.architecture_variant == "diffusion_refine"
    assert prompt_gate.mlflow.experiment_name == "geomaskformer_search"
    assert prompt_gate.mlflow.tags.search_stage == "echo_screen"
    assert "echonet_adult/processed/echonet_png128_10f" in str(adult_cascade.data_path)
    assert adult_cascade.mlflow.tags.dataset == "echonet_adult"
    assert "echonet_pediatric/processed/echonet_pediatric_a4c_png128_10f" in str(pediatric_cascade.data_path)
    assert pediatric_cascade.data.protocol_name == "echonet_pediatric_a4c_endpoint"
    assert pediatric_cascade.mlflow.tags.dataset == "echonet_pediatric"
    assert "camus/processed/camus_png256_10f" in str(camus_cfg.data_path)
    assert camus_fullres_cascade.model.geomaskformer.architecture_variant == "fullres_cascade"
