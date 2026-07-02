from __future__ import annotations

from omegaconf import OmegaConf
import torch

from losses import LossComputer
from model.cardia import CARDIA, CardiacContextEncoder, CardiacKVMemory, GridODESolver, SelectiveLinearDeformationMemory
from models.registry import MODEL_REGISTRY
from utils.model_capacity import infer_unext_capacity


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
                "stage3_injection_learnable": True,
                "runtime_token_dim": 8,
                "use_cardiac_context": True,
                "cardiac_context_hidden_dim": 16,
                "cardiac_context_gate_init": 0.35,
                "dynamic_context_trust_floor": 0.35,
                "runtime_logit_fusion": {
                    "enabled": True,
                    "hidden_dim": 16,
                    "init_biases": [1.0, 0.8, -0.2, -0.2, -0.6],
                },
                "memory_type": "kv",
                "deformation_source": "memory",
                "kv_memory": {
                    "key_dim": 8,
                    "stage3_value_dim": 12,
                    "stage2_value_dim": 10,
                    "hidden_dim": 16,
                    "write_bias": -1.0,
                    "decay_bias": 1.0,
                    "reliability_floor": 0.05,
                },
                "sldm": {
                    "key_dim": 8,
                    "value_dim": 8,
                    "zero_init": True,
                    "use_rmsnorm": True,
                    "forget_bias": 1.0,
                    "write_bias": -1.0,
                },
                "stage2_head_scales": [0.5, 1.0, 1.5],
                "proposal_loss": "soft_oracle",
                "lambda_cardia_flow_smooth": 0.002,
                "lambda_cardia_stage3_flow_smooth": 0.003,
                "lambda_cardia_stage2_flow_smooth": 0.0015,
                "lambda_cardia_proposal_top1": 0.05,
                "lambda_cardia_selector_global": 0.025,
                "lambda_cardia_selector_spatial": 0.05,
                "lambda_cardia_selector_margin_global": 0.025,
                "lambda_cardia_selector_margin_spatial": 0.05,
                "lambda_cardia_memory_readout": 0.05,
                "lambda_cardia_memory_readout_stage3": 0.025,
                "lambda_cardia_reliability_write": 0.01,
            },
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


def test_grid_ode_solver_identity_samples_current_anchor():
    solver = GridODESolver(padding_mode="border", align_corners=False)
    anchor_feat_t = torch.randn(2, 4, 8, 8)
    ode_flow_t = torch.zeros(2, 3, 2, 8, 8)
    selector = torch.zeros(2, 3, 8, 8)
    selector[:, 0] = 1.0
    dynamic_anchor_t, solved, aux = solver(anchor_feat_t, ode_flow_t, selector)
    assert solved.shape == (2, 3, 4, 8, 8)
    assert aux["grid_oob_ratio"].shape == (2,)
    assert torch.allclose(dynamic_anchor_t, anchor_feat_t, atol=1.0e-6)


def test_grid_ode_solver_positive_x_flow_is_backward_sampling_left_motion():
    solver = GridODESolver(padding_mode="zeros", align_corners=False)
    anchor_feat_t = torch.zeros(1, 1, 9, 9)
    anchor_feat_t[:, :, 4, 4] = 1.0
    ode_flow_t = torch.zeros(1, 1, 2, 9, 9)
    ode_flow_t[:, :, 0] = 2.0 / 9.0
    selector = torch.ones(1, 1, 9, 9)
    dynamic_anchor_t, _, aux = solver(anchor_feat_t, ode_flow_t, selector)
    assert aux["grid_oob_ratio"].shape == (1,)
    assert dynamic_anchor_t[0, 0, 4, 3] > 0.99
    assert dynamic_anchor_t[0, 0, 4, 4] < 0.01


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
    assert aux["spatial_pooled_selector_logits"].shape == (2, 1, 3)
    assert aux["proposal_spatial_top1_logits"].shape == (2, 1, 64, 64)
    assert aux["proposal_mixture_proxy_logits"].shape == (2, 1, 64, 64)
    assert aux["dynamic_decoder_logits"].shape == (2, 1, 64, 64)
    assert aux["memory_prior_logits"].shape == (2, 1, 64, 64)
    assert aux["runtime_logit_fusion_enabled"].item() == 1.0
    assert aux["logit_fusion_entropy"].shape == (2,)
    assert aux["logit_fusion_temperature"].shape == (1,)
    for name in ("dynamic", "base", "proposal_top1", "proposal_mixture", "memory_prior"):
        assert aux[f"logit_fusion_weight_{name}"].shape == (2,)
    assert torch.allclose(aux["selector_logits"], aux["global_selector_logits"])
    assert torch.allclose(aux["head_weights"], torch.softmax(aux["global_selector_logits"], dim=-1))
    assert aux["stage3_gamma"].item() >= 0.0
    assert aux["stage2_gamma"].item() >= 0.0
    assert aux["boundary_gamma"].item() >= 0.0
    assert aux["runtime_state_detached"].item() == 1.0
    assert aux["memory_type_kv"].item() == 1.0
    assert aux["deformation_source_memory"].item() == 1.0
    assert aux["context_enabled"].item() == 1.0
    assert aux["context_area"].shape == (2,)
    assert aux["context_delta_area"].shape == (2,)
    assert aux["context_delta_centroid_abs"].shape == (2,)
    assert aux["context_token_rms"].shape == (2,)
    assert aux["stage3_context_gate"].item() >= 0.0
    assert aux["stage2_context_gate"].item() >= 0.0
    assert aux["dynamic_context_trust"].shape == (2,)
    assert aux["stage3_dynamic_trust_mean"].shape == (2,)
    assert aux["stage2_dynamic_trust_mean"].shape == (2,)
    assert torch.all(aux["dynamic_context_trust"] >= 0.35)
    assert aux["stage3_decay_mean"].shape == (2,)
    assert aux["stage3_dynamic_anchor_minus_anchor_abs_mean"].shape == (2,)
    assert aux["stage3_grid_oob_ratio"].shape == (2,)
    assert aux["stage3_fusion_gate_p05"].shape == (2,)
    assert aux["stage3_runtime_state_abs_mean"].shape == (2,)
    assert aux["stage3_runtime_state_rms"].shape == (2,)
    assert aux["stage3_runtime_token_abs_mean"].shape == (2,)
    assert aux["stage3_memory_reliability"].shape == (2,)
    assert aux["stage3_memory_write_mean"].shape == (2,)
    assert aux["stage3_memory_decay_mean"].shape == (2,)
    assert aux["stage3_memory_read_gate_mean"].shape == (2,)
    assert aux["stage3_memory_mask_prior_logits"].shape[-2:] == (8, 8)
    assert aux["stage3_injection_scale"].shape == (1,)
    assert aux["stage2_dynamic_anchor_minus_anchor_abs_mean"].shape == (2,)
    assert aux["stage2_grid_oob_ratio"].shape == (2,)
    assert aux["stage2_delta_proj_abs_mean"].shape == (2,)
    assert aux["stage2_head_usage"].shape == (2, 3)
    assert aux["stage2_global_head_usage"].shape == (2, 3)
    assert aux["stage2_spatial_head_usage"].shape == (2, 3)
    assert aux["stage2_global_spatial_agreement"].shape == (2,)
    assert aux["stage2_memory_reliability"].shape == (2,)
    assert aux["stage2_memory_write_mean"].shape == (2,)
    assert aux["stage2_memory_decay_mean"].shape == (2,)
    assert aux["stage2_memory_read_gate_mean"].shape == (2,)
    assert aux["stage2_memory_mask_prior_logits"].shape[-2:] == (16, 16)
    assert aux["boundary_edge_gate"].shape == (2, 1, 64, 64)
    assert aux["boundary_delta_map"].shape[0] == 2
    assert torch.allclose(torch.sigmoid(aux["boundary_logits"]), aux["boundary_edge_gate"], atol=1.0e-6)
    assert aux["boundary_edge_effective_mean"].shape == (2,)
    assert aux["boundary_edge_gate_mean"].shape == (2,)
    for key in (
        "stage2_offset_px_mean",
        "stage2_offset_px_p95",
        "stage2_flow_smooth",
        "stage2_dynamic_anchor_minus_anchor_abs_mean",
        "stage2_fused_minus_anchor_abs_mean",
        "stage2_write_mean",
        "stage2_head_entropy",
        "stage2_head_usage_entropy",
        "stage3_offset_px_mean",
        "stage3_offset_px_p95",
        "stage3_flow_smooth",
        "stage3_dynamic_anchor_minus_anchor_abs_mean",
        "stage3_fused_minus_anchor_abs_mean",
        "stage3_write_mean",
        "stage3_decay_mean",
        "proposal_top1_logits",
        "proposal_logits",
        "final_minus_base_logit_abs_mean",
        "boundary_edge_gate_p05",
        "boundary_edge_gate_p95",
    ):
        assert key in aux
        assert torch.is_tensor(aux[key])


def test_sldm_shape_and_detached_state_contract():
    sldm = SelectiveLinearDeformationMemory(
        8,
        key_dim=4,
        value_dim=6,
        runtime_token_dim=5,
        forget_bias=1.0,
        write_bias=-1.0,
    )
    anchor = torch.randn(2, 8, 8, 8, requires_grad=True)
    area = torch.randn(2, 2)
    context, state, token, aux = sldm(anchor, None, area)
    assert context.shape == anchor.shape
    assert state.shape == (2, 6, 4)
    assert token.shape == (2, 5)
    assert aux["sldm_write_mean"].shape == (2,)
    assert aux["sldm_forget_mean"].shape == (2,)
    next_context, next_state, _, _ = sldm(anchor, state.detach(), area, token.detach())
    assert next_context.shape == anchor.shape
    assert next_state.grad_fn is not None
    assert torch.isfinite(next_state).all()


def test_cardiac_kv_memory_uses_mask_conditioned_state_and_gates():
    memory = CardiacKVMemory(
        8,
        key_dim=4,
        value_dim=6,
        runtime_token_dim=5,
        hidden_dim=12,
        write_bias=-1.0,
        decay_bias=1.0,
        reliability_floor=0.05,
    )
    anchor = torch.randn(2, 8, 8, 8, requires_grad=True)
    logits = torch.full((2, 1, 32, 32), -5.0)
    logits[:, :, 8:24, 10:22] = 5.0
    area = torch.randn(2, 2)
    context, state, token, aux = memory(anchor, None, logits, area)
    assert context.shape == anchor.shape
    assert state["key"].shape == (2, 4)
    assert state["value"].shape == (2, 6, 8, 8)
    assert state["mask"].shape == (2, 1, 8, 8)
    assert token.shape == (2, 5)
    assert aux["memory_reliability"].shape == (2,)
    assert aux["memory_write_mean"].shape == (2,)
    assert aux["memory_decay_mean"].shape == (2,)
    assert aux["memory_read_gate_mean"].shape == (2,)
    assert aux["memory_mask_prior_logits"].shape == (2, 1, 8, 8)
    assert torch.all(aux["memory_write_mean"] >= 0.0)
    context2, state2, token2, aux2 = memory(anchor, {k: v.detach() for k, v in state.items()}, logits, area, token.detach())
    assert context2.shape == anchor.shape
    assert state2["value"].grad_fn is not None
    assert token2.shape == token.shape
    assert aux2["memory_current_agreement"].shape == (2,)


def test_cardiac_context_encoder_tracks_shape_observation():
    encoder = CardiacContextEncoder(token_dim=6, hidden_dim=12, detach_observation=True)
    logits = torch.full((2, 1, 16, 16), -6.0)
    logits[:, :, 4:12, 5:11] = 6.0
    token, obs, aux = encoder(logits, None, None)
    assert token.shape == (2, 6)
    assert obs.shape == (2, 10)
    assert aux["context_area"].shape == (2,)
    assert torch.all(aux["context_area"] > 0.1)
    shifted = torch.full((2, 1, 16, 16), -6.0)
    shifted[:, :, 4:12, 7:13] = 6.0
    token2, obs2, aux2 = encoder(shifted, token.detach(), obs.detach())
    assert token2.shape == token.shape
    assert torch.all(aux2["context_delta_centroid_abs"] > 0.0)


def test_registry_builds_cardia_aliases():
    cfg = OmegaConf.create({"model": _cfg()})
    for alias in ("cardia", "unext_cardia"):
        cfg.model.name = alias
        model = MODEL_REGISTRY.build(cfg, device=torch.device("cpu"))
        assert isinstance(model, CARDIA)


def test_cardia_stage3_dynamic_path_has_gradient_to_final_logits():
    model = CARDIA(_cfg())
    data = {
        "rgb": torch.rand(1, 2, 1, 64, 64),
        "info": {"num_objects": torch.ones(1, dtype=torch.long)},
    }
    out = model(data)
    loss = out["logits_1"][:, 1].mean()
    loss.backward()
    grad = model.ode_gen3.offset_head.weight.grad
    assert grad is not None
    assert float(grad.abs().sum().item()) > 0.0


def test_cardia_final_logits_gradients_reach_stage2_ode_and_boundary():
    model = CARDIA(_cfg())
    data = {
        "rgb": torch.rand(1, 2, 1, 64, 64),
        "info": {"num_objects": torch.ones(1, dtype=torch.long)},
    }
    out = model(data)
    loss = out["logits_1"][:, 1].mean()
    loss.backward()
    for param in (
        model.ode_gen2.offset_head.weight,
        model.ode_gen2.context_token_proj[-1].weight,
        model.cardiac_context.obs_proj[-1].weight,
        model.boundary_fusion.edge_gate_head.weight,
        model.boundary_fusion.delta_proj.weight,
    ):
        assert param.grad is not None
        assert float(param.grad.abs().sum().item()) > 0.0


def test_cardia_dynamic_and_boundary_delta_proj_are_small_nonzero_init():
    model = CARDIA(_cfg())
    assert float(model.fuse3.delta_proj.weight.abs().sum().item()) > 0.0
    assert float(model.fuse2.delta_proj.weight.abs().sum().item()) > 0.0
    assert float(model.boundary_fusion.delta_proj.weight.abs().sum().item()) > 0.0


def test_cardia_flow_smooth_raw_and_weighted_loss_fields_exist():
    model = CARDIA(_cfg())
    data = {
        "rgb": torch.rand(1, 2, 1, 64, 64),
        "cls_gt": (torch.rand(1, 2, 1, 64, 64) > 0.6).long(),
        "info": {"num_objects": torch.ones(1, dtype=torch.long)},
    }
    data.update(model({"rgb": data["rgb"], "info": data["info"]}))
    losses = LossComputer(OmegaConf.create({"model": _cfg()}), _stage_cfg()).compute(data, [1])
    assert "raw_cardia_stage2_flow_smooth" in losses
    assert "raw_cardia_stage3_flow_smooth" in losses
    assert "weighted_cardia_stage2_flow_smooth" in losses
    assert "weighted_cardia_stage3_flow_smooth" in losses
    assert "aux_cardia_flow_smooth" in losses
    assert "raw_cardia_proposal_top1" in losses
    assert "raw_cardia_selector_global" in losses
    assert "raw_cardia_selector_spatial" in losses
    assert "raw_cardia_selector_margin_global" in losses
    assert "raw_cardia_selector_margin_spatial" in losses
    assert "raw_cardia_memory_readout_stage2" in losses
    assert "raw_cardia_memory_readout_stage3" in losses
    assert "raw_cardia_reliability_write" in losses


def test_cardia_n2_and_lite_parameter_budgets():
    n2 = OmegaConf.create({"model": _cfg()})
    n2.model.cardia.base_dim = 120
    n2.model.cardia.value_dim = 256
    n2.model.cardia.hidden_dim = 256
    n2.model.cardia.runtime_token_dim = 32
    n2.model.cardia.stage2_num_heads = 4
    n2.model.cardia.stage2_head_scales = [0.4, 0.8, 1.2, 1.8]
    n2.model.cardia.backbone = {"name": "official", "official": {"mlp_expansion": 2.0, "latent_blocks": 2, "decoder_mlp_blocks": 1}}
    n2.model.cardia.kv_memory = {
        "key_dim": 64,
        "stage3_value_dim": 240,
        "stage2_value_dim": 120,
        "hidden_dim": 128,
        "write_bias": -1.0,
        "decay_bias": 1.0,
        "reliability_floor": 0.05,
    }
    n2_model = CARDIA(n2.model)
    n2_capacity = infer_unext_capacity(n2_model, n2)
    assert 14.7 < n2_capacity["parameters_m_total"] < 15.2
    assert n2_capacity["backbone_name"] == "official"

    lite = OmegaConf.create(n2)
    lite.model.cardia.base_dim = 108
    lite_model = CARDIA(lite.model)
    lite_capacity = infer_unext_capacity(lite_model, lite)
    assert 12.1 < lite_capacity["parameters_m_total"] < 12.7
