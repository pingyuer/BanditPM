from __future__ import annotations

import logging

import torch

from training.metrics import grad_norm_for_prefixes

log = logging.getLogger(__name__)


def log_final_metrics(trainer, metrics, mode, it, epoch):
    log_items = []
    for k, v in metrics.items():
        log_items.append(f"{k.upper()}={v:.4f}")

    log_str = f"[{mode.capitalize()}] Iter={it} | " + " | ".join(log_items)
    trainer.log.info(log_str)
    if metrics.get("faf/final_below_base_alert", 0.0) >= 1.0:
        trainer.log.warning(
            "FAF final_dice is below base_dice by more than 0.02; "
            "inspect affine mixture safety before extending this run."
        )

    logger = getattr(trainer, "mlflow_logger", None)
    if logger is not None:
        logger.log_eval_summary(metrics, mode=mode, step=it)


def log_train_metrics(trainer, losses, total_loss, it):
    try:
        log_dict = {
            "total_loss": total_loss,
            "lr": trainer.scheduler.get_last_lr()[0],
        }
        model_name = str(trainer.cfg.get("model", {}).get("name", "")).lower()
        if model_name == "cardia":
            active_lambda_prefixes = ("lambda_cardia_",)
        elif model_name in ("unext_gar", "unext_gar_grid_v2", "unext_gar_gridv2"):
            active_lambda_prefixes = ("lambda_gar_",)
        elif model_name in ("faf", "unext_ode_affine", "unext_ode_affine_echo"):
            active_lambda_prefixes = ("lambda_faf_",)
        elif model_name == "functional_anchor":
            active_lambda_prefixes = ("lambda_functional_anchor_",)
        elif model_name in {"rebel", "resampled_belief"}:
            active_lambda_prefixes = ("lambda_rebel_",)
        elif model_name == "debel":
            active_lambda_prefixes = ("lambda_debel_",)
        else:
            active_lambda_prefixes = (
                "lambda_cardia_",
                "lambda_gar_",
                "lambda_faf_",
                "lambda_functional_anchor_",
                "lambda_debel_",
            )
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                log_dict[k] = v.item()
        for attr, name in (
            ("lambda_functional_anchor_anchor", "anchor"),
            ("lambda_functional_anchor_base", "base"),
            ("lambda_functional_anchor_residual_l1", "residual_l1"),
            ("lambda_functional_anchor_boundary", "boundary_residual"),
            ("lambda_functional_anchor_phase", "phase_consistency"),
            ("lambda_functional_anchor_temp", "anchor_temporal"),
            ("lambda_functional_anchor_slot_order", "slot_area_order"),
            ("lambda_functional_anchor_phase_slot", "phase_slot_correlation"),
            ("lambda_functional_anchor_trust_l1", "trust_l1"),
            ("lambda_functional_anchor_trust_entropy", "trust_entropy"),
            ("lambda_functional_anchor_ode_raw_delta", "ode_raw_delta"),
            ("lambda_faf_mixture", "mixture"),
            ("lambda_faf_oracle", "oracle"),
            ("lambda_faf_top1", "top1"),
            ("lambda_faf_selector", "selector"),
            ("lambda_faf_confidence", "confidence"),
            ("lambda_faf_base", "base"),
            ("lambda_faf_coverage", "coverage"),
            ("lambda_faf_sparse", "sparse"),
            ("lambda_faf_diversity", "diversity"),
            ("lambda_faf_temporal", "temporal"),
            ("lambda_faf_write", "write"),
            ("lambda_faf_residual_smallness", "residual_smallness"),
            ("lambda_faf_affine", "affine"),
            ("lambda_faf_velocity", "velocity"),
            ("lambda_gar_base", "base"),
            ("lambda_gar_proposal_oracle", "proposal_oracle"),
            ("lambda_gar_selector", "selector"),
            ("lambda_gar_flow_smooth", "flow_smooth"),
            ("lambda_gar_boundary_aux", "boundary_aux"),
            ("lambda_cardia_base", "base"),
            ("lambda_cardia_proposal_oracle", "proposal_oracle"),
            ("lambda_cardia_proposal_top1", "proposal_top1"),
            ("lambda_cardia_multi_head_fused", "multi_head_fused"),
            ("lambda_cardia_selector", "selector"),
            ("lambda_cardia_selector_margin", "selector_margin"),
            ("lambda_cardia_flow_smooth", "flow_smooth"),
            ("lambda_cardia_stage3_flow_smooth", "stage3_flow_smooth"),
            ("lambda_cardia_stage2_flow_smooth", "stage2_flow_smooth"),
            ("lambda_cardia_boundary_aux", "boundary_aux"),
            ("lambda_cardia_memory_readout", "memory_readout"),
            ("lambda_cardia_memory_readout_stage3", "memory_readout_stage3"),
            ("lambda_cardia_reliability_write", "reliability_write"),
            ("lambda_head_diversity", "head_diversity"),
            ("lambda_rebel_final", "final"),
            ("lambda_rebel_base_aux", "base_aux"),
            ("lambda_rebel_belief_prior", "belief_prior"),
            ("lambda_rebel_obs_aux", "obs_aux"),
            ("lambda_rebel_rebel_aux", "rebel_aux"),
            ("lambda_rebel_corrected_aux", "corrected_aux"),
            ("lambda_rebel_candidate_oracle", "candidate_oracle"),
            ("lambda_rebel_arbitration", "arbitration"),
            ("lambda_rebel_correction", "correction"),
            ("lambda_rebel_temporal", "temporal"),
            ("lambda_rebel_offset_smooth", "offset_smooth"),
            ("lambda_rebel_write_reg", "write_reg"),
            ("lambda_debel_final", "final"),
            ("lambda_debel_anchor", "anchor"),
            ("lambda_debel_grid", "grid"),
            ("lambda_debel_smooth", "smooth"),
            ("lambda_debel_temp", "temp"),
            ("lambda_debel_area", "area"),
            ("lambda_debel_residual", "residual"),
        ):
            if not attr.startswith(active_lambda_prefixes):
                continue
            if hasattr(trainer.loss_computer, attr):
                if attr.startswith("lambda_faf_"):
                    prefix = "lambda_faf"
                elif attr.startswith("lambda_gar_"):
                    prefix = "lambda_gar"
                elif attr.startswith("lambda_cardia_"):
                    prefix = "lambda_cardia"
                elif attr.startswith("lambda_rebel_"):
                    prefix = "lambda_rebel"
                elif attr.startswith("lambda_debel_"):
                    prefix = "lambda_debel"
                else:
                    prefix = "lambda_functional_anchor"
                log_dict[f"{prefix}_{name}"] = getattr(trainer.loss_computer, attr)
        for group in trainer.optimizer.param_groups:
            name = group.get("name")
            if name == "functional_anchor_residual_heads":
                log_dict["residual_head_lr"] = group.get("lr", 0.0)
        if str(trainer.cfg.get("model", {}).get("name", "")).lower() == "functional_anchor":
            model = trainer.model_without_ddp
            if hasattr(model, "_anchor_temperature"):
                temp = model._anchor_temperature(trainer.device, torch.float32)
                log_dict["anchor_temperature"] = float(temp.detach().item())
            if hasattr(model, "_residual_scale_at"):
                scale = model._residual_scale_at({"current_iter": it}, trainer.device, torch.float32)
                log_dict["residual_scale"] = float(scale.detach().item())
        logger = getattr(trainer, "mlflow_logger", None)
        if logger is not None:
            logger.log_train_step(log_dict, step=it)
    except Exception:
        pass


def log_cardia_stats(trainer, data, it: int) -> None:
    memory_keys = sorted(k for k in data.keys() if k.startswith("memory_aux_"))
    if not memory_keys:
        return
    try:
        buckets = {
            "stage3_flow_smooth": [],
            "stage3_offset_px_mean": [],
            "stage3_offset_px_p95": [],
            "stage3_write_mean": [],
            "stage3_decay_mean": [],
            "stage3_context_gate": [],
            "stage3_dynamic_trust_mean": [],
            "stage3_gamma": [],
            "stage3_fusion_gate_mean": [],
            "stage3_fusion_gate_p05": [],
            "stage3_fusion_gate_p95": [],
            "stage3_dynamic_anchor_minus_anchor_abs_mean": [],
            "stage3_fused_minus_anchor_abs_mean": [],
            "stage3_delta_proj_abs_mean": [],
            "stage3_injected_minus_base_abs_mean": [],
            "stage3_grid_oob_ratio": [],
            "stage3_injection_scale": [],
            "stage3_runtime_update_mean": [],
            "stage3_runtime_reset_mean": [],
            "stage3_runtime_state_norm": [],
            "stage3_runtime_state_abs_mean": [],
            "stage3_runtime_state_rms": [],
            "stage3_runtime_token_abs_mean": [],
            "stage3_runtime_token_rms": [],
            "stage3_runtime_token_update_mean": [],
            "stage3_memory_reliability": [],
            "stage3_memory_write_mean": [],
            "stage3_memory_decay_mean": [],
            "stage3_memory_read_gate_mean": [],
            "stage3_memory_current_agreement": [],
            "stage3_memory_boundary_quality": [],
            "stage3_memory_area_ok": [],
            "stage3_memory_readout_abs_mean": [],
            "stage3_sldm_memory_norm_mean": [],
            "stage3_sldm_memory_norm_p95": [],
            "stage3_sldm_update_norm_mean": [],
            "stage3_sldm_forget_mean": [],
            "stage3_sldm_write_mean": [],
            "stage3_sldm_read_abs_mean": [],
            "stage3_sldm_delta_abs_mean": [],
            "stage3_global_selector_entropy": [],
            "stage3_global_spatial_agreement": [],
            "stage2_flow_smooth": [],
            "stage2_offset_px_mean": [],
            "stage2_offset_px_p95": [],
            "stage2_write_mean": [],
            "stage2_decay_mean": [],
            "stage2_context_gate": [],
            "stage2_dynamic_trust_mean": [],
            "stage2_gamma": [],
            "stage2_fusion_gate_mean": [],
            "stage2_fusion_gate_p05": [],
            "stage2_fusion_gate_p95": [],
            "stage2_dynamic_anchor_minus_anchor_abs_mean": [],
            "stage2_fused_minus_anchor_abs_mean": [],
            "stage2_delta_proj_abs_mean": [],
            "stage2_grid_oob_ratio": [],
            "stage2_selector_logit_scale": [],
            "stage2_global_selector_entropy": [],
            "stage2_head_entropy": [],
            "stage2_head_usage_entropy": [],
            "stage2_global_spatial_agreement": [],
            "stage2_runtime_update_mean": [],
            "stage2_runtime_reset_mean": [],
            "stage2_runtime_state_norm": [],
            "stage2_runtime_state_abs_mean": [],
            "stage2_runtime_state_rms": [],
            "stage2_runtime_token_abs_mean": [],
            "stage2_runtime_token_rms": [],
            "stage2_runtime_token_update_mean": [],
            "stage2_memory_reliability": [],
            "stage2_memory_write_mean": [],
            "stage2_memory_decay_mean": [],
            "stage2_memory_read_gate_mean": [],
            "stage2_memory_current_agreement": [],
            "stage2_memory_boundary_quality": [],
            "stage2_memory_area_ok": [],
            "stage2_memory_readout_abs_mean": [],
            "stage2_sldm_memory_norm_mean": [],
            "stage2_sldm_memory_norm_p95": [],
            "stage2_sldm_update_norm_mean": [],
            "stage2_sldm_forget_mean": [],
            "stage2_sldm_write_mean": [],
            "stage2_sldm_read_abs_mean": [],
            "stage2_sldm_delta_abs_mean": [],
            "boundary_gamma": [],
            "boundary_edge_gate_mean": [],
            "boundary_edge_effective_mean": [],
            "boundary_edge_gate_p05": [],
            "boundary_edge_gate_p95": [],
            "boundary_channel_gate_mean": [],
            "boundary_delta_abs_mean": [],
            "boundary_delta_on_band_mean": [],
            "boundary_delta_off_band_mean": [],
            "boundary_delta_band_ratio": [],
            "boundary_edge_gate_on_band_mean": [],
            "boundary_edge_gate_off_band_mean": [],
            "final_minus_base_logit_abs_mean": [],
            "runtime_logit_fusion_enabled": [],
            "logit_fusion_temperature": [],
            "logit_fusion_entropy": [],
            "logit_fusion_fused_minus_base_abs_mean": [],
            "logit_fusion_fused_minus_dynamic_abs_mean": [],
            "logit_fusion_weight_dynamic": [],
            "logit_fusion_weight_base": [],
            "logit_fusion_weight_proposal_top1": [],
            "logit_fusion_weight_proposal_mixture": [],
            "logit_fusion_weight_memory_prior": [],
            "runtime_state_detached": [],
            "memory_type_kv": [],
            "deformation_source_memory": [],
            "context_enabled": [],
            "context_area": [],
            "context_delta_area": [],
            "context_centroid_x": [],
            "context_centroid_y": [],
            "context_delta_centroid_abs": [],
            "context_scale_x": [],
            "context_scale_y": [],
            "context_boundary_energy": [],
            "context_uncertainty": [],
            "context_token_rms": [],
            "context_update_mean": [],
            "dynamic_context_trust": [],
            "cross_attn_entropy": [],
            "cross_attn_gamma": [],
            "cross_attn_weight_std": [],
            "cross_attn_residual_abs_mean": [],
            "use_multi_head_fusion": [],
        }
        for key in memory_keys:
            aux = data.get(key)
            cardia = aux.get("cardia_aux") if isinstance(aux, dict) else None
            if not isinstance(cardia, dict):
                continue
            for name in buckets:
                value = cardia.get(name)
                if torch.is_tensor(value):
                    buckets[name].append(value.float().detach().flatten())
            for stage in ("stage2", "stage3"):
                for usage_name in ("head_usage", "global_head_usage", "spatial_head_usage"):
                    usage = cardia.get(f"{stage}_{usage_name}")
                    if torch.is_tensor(usage):
                        usage_mean = usage.float().detach().mean(dim=0)
                        for idx, value in enumerate(usage_mean.flatten()):
                            buckets.setdefault(f"{stage}_{usage_name}_{idx}", []).append(value.reshape(1))
            if "cls_gt" in data:
                try:
                    ti = int(key.rsplit("_", 1)[-1])
                    gt = (data["cls_gt"][:, ti].float() > 0).float()
                    edge = cardia.get("boundary_edge_gate")
                    delta = cardia.get("boundary_delta_map")
                    if torch.is_tensor(edge) and torch.is_tensor(delta):
                        if gt.dim() == 3:
                            gt = gt.unsqueeze(1)
                        if gt.shape[-2:] != edge.shape[-2:]:
                            gt = torch.nn.functional.interpolate(gt, size=edge.shape[-2:], mode="nearest")
                        dil = torch.nn.functional.max_pool2d(gt, kernel_size=3, stride=1, padding=1)
                        ero = 1.0 - torch.nn.functional.max_pool2d(1.0 - gt, kernel_size=3, stride=1, padding=1)
                        band = (dil - ero).clamp(0.0, 1.0).bool()
                        edge_item = edge.detach().float()
                        delta_item = delta.detach().abs().float().mean(dim=1, keepdim=True)
                        on_delta = delta_item[band]
                        off_delta = delta_item[~band]
                        on_edge = edge_item[band]
                        off_edge = edge_item[~band]
                        if on_delta.numel() > 0 and off_delta.numel() > 0:
                            buckets["boundary_delta_on_band_mean"].append(on_delta.mean().reshape(1))
                            buckets["boundary_delta_off_band_mean"].append(off_delta.mean().reshape(1))
                            buckets["boundary_delta_band_ratio"].append((on_delta.mean() / off_delta.mean().clamp_min(1.0e-6)).reshape(1))
                            buckets["boundary_edge_gate_on_band_mean"].append(on_edge.mean().reshape(1))
                            buckets["boundary_edge_gate_off_band_mean"].append(off_edge.mean().reshape(1))
                except Exception:
                    pass
        _ode = {
            "stage2_flow_smooth", "stage2_offset_px_mean", "stage2_offset_px_p95",
            "stage2_grid_oob_ratio",
            "stage3_flow_smooth", "stage3_offset_px_mean", "stage3_offset_px_p95",
            "stage3_grid_oob_ratio",
        }
        _selector = {
            "stage2_head_usage_entropy", "stage2_selector_logit_scale",
            "stage2_global_selector_entropy", "stage2_global_spatial_agreement",
            "stage2_head_entropy",
            "stage3_global_selector_entropy", "stage3_global_spatial_agreement",
        }
        _selector_prefixes = ("head_usage_", "global_head_usage_", "spatial_head_usage_")
        _memory = {
            "stage2_runtime_update_mean", "stage2_runtime_reset_mean",
            "stage2_runtime_state_norm", "stage2_runtime_state_abs_mean",
            "stage2_runtime_state_rms", "stage2_runtime_token_abs_mean",
            "stage2_runtime_token_rms", "stage2_runtime_token_update_mean",
            "stage2_write_mean", "stage2_decay_mean",
            "stage2_sldm_memory_norm_mean", "stage2_sldm_memory_norm_p95",
            "stage2_sldm_update_norm_mean", "stage2_sldm_forget_mean",
            "stage2_sldm_write_mean", "stage2_sldm_read_abs_mean",
            "stage2_sldm_delta_abs_mean",
            "stage3_runtime_update_mean", "stage3_runtime_reset_mean",
            "stage3_runtime_state_norm", "stage3_runtime_state_abs_mean",
            "stage3_runtime_state_rms", "stage3_runtime_token_abs_mean",
            "stage3_runtime_token_rms", "stage3_runtime_token_update_mean",
            "stage3_write_mean", "stage3_decay_mean",
            "stage3_sldm_memory_norm_mean", "stage3_sldm_memory_norm_p95",
            "stage3_sldm_update_norm_mean", "stage3_sldm_forget_mean",
            "stage3_sldm_write_mean", "stage3_sldm_read_abs_mean",
            "stage3_sldm_delta_abs_mean",
        }
        _injection = {
            "stage3_injection_scale", "stage3_injected_minus_base_abs_mean",
            "cross_attn_entropy", "cross_attn_gamma", "cross_attn_weight_std",
            "cross_attn_residual_abs_mean",
        }
        _fusion = {
            "stage2_gamma", "stage2_fusion_gate_mean",
            "stage2_fusion_gate_p05", "stage2_fusion_gate_p95",
            "stage2_delta_proj_abs_mean",
            "stage2_dynamic_anchor_minus_anchor_abs_mean",
            "stage2_fused_minus_anchor_abs_mean",
            "stage3_gamma", "stage3_fusion_gate_mean",
            "stage3_fusion_gate_p05", "stage3_fusion_gate_p95",
            "stage3_delta_proj_abs_mean",
            "stage3_dynamic_anchor_minus_anchor_abs_mean",
            "stage3_fused_minus_anchor_abs_mean",
        }
        _boundary = {
            "boundary_gamma", "boundary_edge_gate_mean",
            "boundary_edge_effective_mean",
            "boundary_edge_gate_p05", "boundary_edge_gate_p95",
            "boundary_channel_gate_mean", "boundary_delta_abs_mean",
            "boundary_delta_on_band_mean", "boundary_delta_off_band_mean",
            "boundary_delta_band_ratio",
            "boundary_edge_gate_on_band_mean", "boundary_edge_gate_off_band_mean",
        }
        _new_group_prefixes = {
            "ode": "cardia/ode",
            "selector": "cardia/selector",
            "memory": "cardia/memory",
            "injection": "cardia/injection",
            "fusion": "cardia/fusion",
            "boundary": "cardia/boundary",
        }
        metrics = {}
        for name, tensors in buckets.items():
            if not tensors:
                continue
            value = torch.cat(tensors).mean().item()
            if name in _boundary:
                new_log_name = f"cardia/boundary/{name.removeprefix('boundary_')}"
            elif name.startswith("stage"):
                stage, metric = name.split("_", 1)
                if name in _ode:
                    new_log_name = f"cardia/ode/{stage}/{metric}"
                elif name in _selector:
                    new_log_name = f"cardia/selector/{stage}/{metric}"
                elif any(name.startswith(f"{stage}_{p}") for p in _selector_prefixes):
                    new_log_name = f"cardia/selector/{stage}/{metric}"
                elif name in _memory:
                    if metric.startswith("sldm_"):
                        new_log_name = f"cardia/memory/{stage}/sldm/{metric.removeprefix('sldm_')}"
                    else:
                        new_log_name = f"cardia/memory/{stage}/{metric}"
                elif name in _fusion:
                    new_log_name = f"cardia/fusion/{stage}/{metric}"
                elif name in _injection:
                    new_log_name = f"cardia/injection/{metric}"
                else:
                    new_log_name = f"cardia/{stage}/{metric}"
            elif name in _injection:
                new_log_name = f"cardia/injection/{name}"
            else:
                new_log_name = f"cardia/{name}"
            if name.startswith("stage"):
                stage, metric = name.split("_", 1)
                if metric.startswith("sldm_"):
                    old_log_name = f"cardia/{stage}/sldm/{metric.removeprefix('sldm_')}"
                else:
                    old_log_name = f"cardia/{stage}/{metric}"
            elif name.startswith("boundary_"):
                old_log_name = f"cardia/boundary/{name.removeprefix('boundary_')}"
            else:
                old_log_name = f"cardia/{name}"
            if name in {"stage2_offset_px_mean", "stage3_offset_px_mean"}:
                metrics.setdefault("solver_offset_px_mean", value)
            if name in {"stage2_offset_px_p95", "stage3_offset_px_p95"}:
                metrics.setdefault("solver_offset_px_p95", value)
            trainer.log.log_scalar(new_log_name, value, it)
            if old_log_name != new_log_name:
                trainer.log.log_scalar(old_log_name, value, it)
            metrics[name] = value
        if metrics:
            logger = getattr(trainer, "mlflow_logger", None)
            if logger is not None:
                if hasattr(logger, "log_cardia_diagnostics"):
                    logger.log_cardia_diagnostics(metrics, step=it)
                else:
                    logger.log_metrics(metrics, step=it, prefix="cardia")
    except Exception:
        pass


def log_cardia_grad_stats(trainer, it: int) -> None:
    if str(trainer.cfg.get("model", {}).get("name", "")).lower() not in {"cardia", "unext_cardia"}:
        return
    metrics = {}
    for key, prefixes in {
        "cardia/grad/offset_head_norm": ("ode_gen2.offset_head.", "ode_gen3.offset_head."),
        "cardia/grad/odegen_norm": ("ode_gen2.", "ode_gen3."),
        "cardia/grad/delta_proj_norm": ("fuse2.delta_proj.", "fuse3.delta_proj.", "boundary_fusion.delta_proj."),
        "cardia/grad/sldm_norm": ("sldm2.", "sldm3."),
        "cardia/grad/logit_fusion_norm": ("logit_fusion.",),
    }.items():
        value = grad_norm_for_prefixes(trainer, prefixes)
        if value is not None:
            metrics[key] = value
    if metrics:
        trainer._log_metrics(metrics, step=it)


def log_rebel_grad_stats(trainer, it: int) -> None:
    if str(trainer.cfg.get("model", {}).get("name", "")).lower() not in {"rebel", "resampled_belief"}:
        return
    metrics = {}
    for key, prefixes in {
        "rebel/grad/ode_offset_norm": ("ode.delta_obs_head.", "ode.delta_mem_head."),
        "rebel/grad/ode_control_norm": ("ode.gate_head.", "ode.write_decay_head."),
        "rebel/grad/memory_norm": ("memory.",),
        "rebel/grad/decoder_norm": ("decoder.",),
        "rebel/grad/fusion_norm": ("fusion.",),
        "rebel/grad/correction_norm": ("correction.",),
    }.items():
        value = grad_norm_for_prefixes(trainer, prefixes)
        if value is not None:
            metrics[key] = value
    if metrics:
        trainer._log_metrics(metrics, step=it)


def log_debel_stats(trainer, data, it: int) -> None:
    if str(trainer.cfg.get("model", {}).get("name", "")).lower() != "debel":
        return
    aux = data.get("aux")
    if not isinstance(aux, dict):
        return
    metrics = {}
    for key, value in aux.items():
        if torch.is_tensor(value):
            metrics[key] = value.detach().float().mean().item()
    if metrics:
        trainer._log_metrics(metrics, step=it)


def log_debel_grad_stats(trainer, it: int) -> None:
    if str(trainer.cfg.get("model", {}).get("name", "")).lower() != "debel":
        return
    metrics = {}
    for key, prefixes in {
        "debel/grad/transformer_norm": ("video_encoder.",),
        "debel/grad/query_norm": ("query_decoder.",),
        "debel/grad/grid_head_norm": ("grid_solver.",),
        "debel/grad/residual_norm": ("boundary_residual.",),
        "debel/grad/frame_anchor_norm": ("frame_net.", "backbone."),
    }.items():
        value = grad_norm_for_prefixes(trainer, prefixes)
        if value is not None:
            metrics[key] = value
    if metrics:
        trainer._log_metrics(metrics, step=it)


def log_rebel_stats(trainer, data, it: int) -> None:
    memory_keys = sorted(k for k in data.keys() if k.startswith("memory_aux_"))
    if not memory_keys:
        return
    try:
        buckets = {}
        for key in memory_keys:
            aux = data[key]
            rebel_aux = aux.get("rebel_aux") if isinstance(aux, dict) else None
            if not isinstance(rebel_aux, dict):
                continue
            mapping = {
                "r_obs_mean": "r_obs",
                "write_fast_mean": "write_fast",
                "write_slow_mean": "write_slow",
                "decay_fast_mean": "decay_fast",
                "decay_slow_mean": "decay_slow",
                "disagreement_mean": "disagreement",
                "memory_prior_area_mean": "memory_prior_area",
                "final_minus_base_abs_mean": "final_minus_base_abs",
                "final_minus_memory_abs_mean": "final_minus_memory_abs",
                "corrected_minus_rebel_abs_mean": "corrected_minus_rebel_abs",
                "belief_feature_delta_norm": "belief_feature_delta_norm",
                "w_mask_delta_mean": "w_mask_delta",
                "s_mask_delta_mean": "s_mask_delta",
                "w_feat_delta_mean": "w_feat_delta",
                "s_feat_delta_mean": "s_feat_delta",
                "offset_obs_px_mean": "offset_obs_px",
                "offset_mem_px_mean": "offset_mem_px",
                "correction_scale": "correction_scale",
                "arbitration_entropy": "arbitration_entropy",
                "arbitration_temperature": "arbitration_temperature",
                "arbitration_weight_base": "arbitration_weight_base",
                "arbitration_weight_obs": "arbitration_weight_obs",
                "arbitration_weight_belief": "arbitration_weight_belief",
                "arbitration_weight_rebel": "arbitration_weight_rebel",
                "arbitration_weight_corrected": "arbitration_weight_corrected",
            }
            for metric_name, aux_key in mapping.items():
                value = rebel_aux.get(aux_key)
                if torch.is_tensor(value):
                    buckets.setdefault(metric_name, []).append(value.detach().float().reshape(-1))
        metrics = {}
        for name, values in buckets.items():
            cat = torch.cat(values)
            metrics[f"rebel/{name}"] = cat.mean().item()
            if name in {"offset_obs_px_mean", "offset_mem_px_mean"}:
                metrics[f"rebel/{name.replace('_mean', '_max')}"] = cat.max().item()
        if "rebel/final_minus_base_abs_mean" in metrics and it > 800 and metrics["rebel/final_minus_base_abs_mean"] < 1.0e-3:
            trainer.log.warning("ReBel collapsed to base path")
        if metrics.get("rebel/write_slow_mean", 0.0) > metrics.get("rebel/write_fast_mean", 1.0):
            trainer.log.warning("stable memory writes faster than working memory")
        if it > 800 and metrics.get("rebel/offset_obs_px_mean", 0.0) == 0.0 and metrics.get("rebel/offset_mem_px_mean", 0.0) == 0.0:
            trainer.log.warning("ODE field collapsed to identity")
        if it < 1000 and metrics.get("rebel/correction_scale", 0.0) > 0.5:
            trainer.log.warning("Correction head may dominate belief decoder")
        for key, value in metrics.items():
            trainer.log.log_scalar(key, value, it)
        if metrics:
            trainer._log_metrics(metrics, step=it)
    except Exception:
        pass
