from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from model.functional_anchor.anchor_bank import FunctionalAnchorBank
from model.functional_anchor.anchor_decoder import AnchorDecoder
from model.functional_anchor.confidence_fusion import ConfidenceFusion
from model.functional_anchor.multilevel_injector import MultiLevelInjector
from model.functional_anchor.phase_encoder import PhaseEncoder
from model.functional_anchor.residual_heads import ResidualHeads
from model.functional_anchor.temporal_state_ode import TemporalStateODE
from model.modules.unext import UNeXtBackbone
from utils.tensor_utils import aggregate


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class FunctionalAnchorSegmenter(nn.Module):
    """Phase-aware functional anchor with UNeXt residual refinement."""

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        method_cfg = _cfg_get(cfg, "functional_anchor", cfg)
        self.in_channels = int(_cfg_get(method_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(method_cfg, "num_classes", 2))
        self.base_dim = int(_cfg_get(method_cfg, "base_dim", 32))
        self.value_dim = int(_cfg_get(method_cfg, "value_dim", 128))
        self.num_slots = int(_cfg_get(method_cfg, "num_slots", 5))
        self.state_dim = int(_cfg_get(method_cfg, "state_dim", 128))
        self.phase_dim = int(_cfg_get(method_cfg, "phase_dim", 32))
        self.hidden_dim = int(_cfg_get(method_cfg, "hidden_dim", 128))
        self.anchor_size = int(_cfg_get(method_cfg, "anchor_size", 32))
        self.residual_clip = float(_cfg_get(method_cfg, "residual_clip", 1.5))
        self.prediction_mode = str(_cfg_get(method_cfg, "prediction_mode", "base_primary"))
        self.use_anchor_features_in_residual = bool(_cfg_get(method_cfg, "use_anchor_features_in_residual", True))
        self.disable_injector = bool(_cfg_get(method_cfg, "disable_injector", False))
        phase_cfg = _cfg_get(method_cfg, "phase", {})
        self.normalized_time_weight = float(_cfg_get(phase_cfg, "normalized_time_weight", 1.0))
        temporal_state_cfg = _cfg_get(method_cfg, "temporal_state", {})
        self.detach_state = bool(_cfg_get(temporal_state_cfg, "detach_state", True))
        self.detach_every = int(_cfg_get(temporal_state_cfg, "detach_every", 1))

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=self.base_dim,
            value_dim=self.value_dim,
        )
        feature_dims = {
            "low": self.base_dim,
            "mid": self.base_dim * 2,
            "high": self.base_dim * 4,
            "dec": self.base_dim,
        }
        evidence_dim = sum(feature_dims.values())
        self.phase_encoder = PhaseEncoder(self.num_slots, self.phase_dim, self.hidden_dim)
        self.state_ode = TemporalStateODE(evidence_dim, self.phase_dim, self.state_dim, self.hidden_dim)
        self.anchor_bank = FunctionalAnchorBank(self.num_slots, self.state_dim, self.phase_dim, self.hidden_dim)
        self.anchor_decoder = AnchorDecoder(
            self.num_slots,
            self.state_dim,
            self.phase_dim,
            feature_dims,
            self.anchor_size,
            self.hidden_dim,
        )
        self.residual_heads = ResidualHeads(
            feature_dims,
            self.hidden_dim,
            self.residual_clip,
            use_anchor_features=self.use_anchor_features_in_residual,
        )
        self.injector = MultiLevelInjector(feature_dims)
        self.fusion = ConfidenceFusion(self.prediction_mode, self.residual_clip)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _pool_evidence(self, feat: dict[str, torch.Tensor], num_objects: int) -> torch.Tensor:
        pooled = []
        for level in ("low", "mid", "high", "decoder_feature"):
            tensor = feat[level if level != "decoder_feature" else "decoder_feature"]
            pooled.append(tensor.mean(dim=(-2, -1)))
        evidence = torch.cat(pooled, dim=-1)
        return evidence.unsqueeze(1).expand(-1, num_objects, -1)

    def _as_tensor(self, value, B: int, device, dtype) -> torch.Tensor | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=dtype)
        elif isinstance(value, (list, tuple)):
            try:
                tensor = torch.tensor(value, device=device, dtype=dtype)
            except Exception:
                return None
        else:
            try:
                tensor = torch.full((B,), float(value), device=device, dtype=dtype)
            except Exception:
                return None
        if tensor.dim() == 0:
            tensor = tensor.view(1).expand(B)
        return tensor.flatten()[:B]

    def _info_value(self, data: Dict, *names: str):
        info = data.get("info", {})
        for name in names:
            if name in data:
                return data[name]
            if isinstance(info, dict) and name in info:
                return info[name]
        return None

    def _metadata_phase(self, data: Dict, ti: int, B: int, T: int, device, dtype) -> torch.Tensor | None:
        ed = self._as_tensor(
            self._info_value(data, "ed_frame", "ed_frame_idx", "ed_index", "ED", "ED_frame"),
            B,
            device,
            dtype,
        )
        es = self._as_tensor(
            self._info_value(data, "es_frame", "es_frame_idx", "es_index", "ES", "ES_frame"),
            B,
            device,
            dtype,
        )
        if ed is None or es is None:
            return None
        frame_indices = data.get("frame_indices")
        if torch.is_tensor(frame_indices) and frame_indices.dim() >= 2 and frame_indices.shape[0] >= B:
            current = frame_indices[:B, ti].to(device=device, dtype=dtype)
        else:
            current = torch.full((B,), float(ti), device=device, dtype=dtype)
        es_span = (es - ed).abs().clamp_min(1.0)
        end_span = (float(T - 1) - es).abs().clamp_min(1.0)
        before_es = current <= es
        phase = torch.where(
            before_es,
            0.5 * ((current - ed).clamp_min(0.0) / es_span),
            0.5 + 0.5 * ((current - es).clamp_min(0.0) / end_span),
        )
        return phase.clamp(0.0, 1.0)

    def _phase_at(
        self,
        data: Dict,
        ti: int,
        B: int,
        N: int,
        T: int,
        device,
        dtype,
        *,
        current_area: torch.Tensor,
        prev_area: torch.Tensor,
        area_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        override = data.get("phase_override")
        if torch.is_tensor(override):
            phase = override[:, ti]
            if phase.dim() == 1:
                phase = phase.unsqueeze(1).expand(-1, N)
            phase = phase.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            return phase, torch.zeros(B, N, device=device, dtype=dtype), torch.ones(B, N, device=device, dtype=dtype)
        metadata_phase = self._metadata_phase(data, ti, B, T, device, dtype)
        if metadata_phase is not None:
            return (
                metadata_phase.unsqueeze(1).expand(-1, N),
                torch.zeros(B, N, device=device, dtype=dtype),
                torch.ones(B, N, device=device, dtype=dtype),
            )
        if ti > 0:
            norm = float(ti) / max(float(T - 1), 1.0)
            contracting_phase = min(0.5, 0.5 * norm / 0.5) if norm <= 0.5 else 0.5
            expanding_phase = 0.5 + 0.5 * norm
            phase = torch.where(
                area_velocity < 0,
                torch.full_like(current_area, contracting_phase),
                torch.full_like(current_area, min(expanding_phase, 1.0)),
            )
            uncertain = area_velocity.abs() < 1.0e-5
            fallback = torch.full_like(current_area, norm)
            phase = torch.where(uncertain, fallback, phase)
            return phase.clamp(0.0, 1.0), torch.ones(B, N, device=device, dtype=dtype), torch.ones(B, N, device=device, dtype=dtype)
        value = float(ti) / max(float(T - 1), 1.0)
        return (
            torch.full((B, N), value, device=device, dtype=dtype),
            torch.full((B, N), 2.0, device=device, dtype=dtype),
            torch.full((B, N), self.normalized_time_weight, device=device, dtype=dtype),
        )

    def _empty_prev(self, B: int, N: int, device, dtype) -> dict[str, torch.Tensor | None]:
        return {
            "z": None,
            "area": torch.zeros(B, N, device=device, dtype=dtype),
            "area_velocity": torch.zeros(B, N, device=device, dtype=dtype),
            "slot_weights": torch.full((B, N, self.num_slots), 1.0 / self.num_slots, device=device, dtype=dtype),
        }

    def _object_logits_to_full(self, object_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.sigmoid(object_logits)
        logits = aggregate(masks, dim=1)
        return logits, torch.softmax(logits, dim=1)[:, 1:]

    def forward(self, data: Dict) -> Dict:
        images = data["rgb"]
        B, T = images.shape[:2]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(max(num_objects), 1)
        prev = self._empty_prev(B, max_num_objects, images.device, images.dtype)
        out: Dict = {"num_objects": num_objects}

        for ti in range(T):
            feat = self.backbone(self._normalize(images[:, ti]))
            feats = {
                "low": feat["low"],
                "mid": feat["mid"],
                "high": feat["high"],
                "dec": feat["decoder_feature"],
            }
            base_object_logits = feat["logits"][:, 1:2].expand(-1, max_num_objects, -1, -1)
            evidence = self._pool_evidence(feat, max_num_objects)
            prev_area = prev["area"]
            prev_velocity = prev["area_velocity"]
            current_area = torch.sigmoid(base_object_logits).mean(dim=(-2, -1)).detach()
            area_velocity = current_area - prev_area.detach()
            area_acceleration = area_velocity - prev_velocity.detach()
            norm_time, phase_source_code, phase_reliability = self._phase_at(
                data,
                ti,
                B,
                max_num_objects,
                T,
                images.device,
                images.dtype,
                current_area=current_area,
                prev_area=prev_area,
                area_velocity=area_velocity,
            )
            phase_embed, phase_descriptor = self.phase_encoder(
                norm_time,
                prev_area=prev_area,
                area_velocity=area_velocity,
                area_acceleration=area_acceleration,
                phase_reliability=phase_reliability,
                slot_history=prev["slot_weights"],
            )
            z, dz = self.state_ode(prev["z"], evidence, phase_embed)
            slot_weights, slot_aux = self.anchor_bank(z, phase_embed, norm_time)
            target_sizes = {level: feats[level].shape[-2:] for level in feats}
            anchor_logits, anchor_features = self.anchor_decoder(
                z,
                phase_embed,
                slot_weights,
                target_sizes,
                base_object_logits.shape[-2:],
            )
            residuals = self.residual_heads(
                feats,
                anchor_logits,
                base_object_logits,
                anchor_features,
                phase_confidence=phase_reliability,
            )
            injected = feats if self.disable_injector else self.injector(feats, anchor_features, residuals)
            residuals = self.residual_heads(
                injected,
                anchor_logits,
                base_object_logits,
                anchor_features,
                phase_confidence=phase_reliability,
            )
            final_object_logits, fusion_aux = self.fusion(
                anchor_logits=anchor_logits,
                base_logits=base_object_logits,
                shape_residual=residuals["shape_residual_logits"],
                boundary_residual=residuals["boundary_residual_logits"],
                anchor_trust=residuals["anchor_trust"],
            )
            logits, masks = self._object_logits_to_full(final_object_logits)
            anchor_area = torch.sigmoid(anchor_logits).mean(dim=(-2, -1)).detach()
            next_area_velocity = anchor_area - prev_area.detach()
            out[f"functional_anchor_area_{ti}"] = anchor_area
            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = masks
            out[f"aux_{ti}"] = {
                "base_foreground_logits": base_object_logits.detach(),
                "object_logits": final_object_logits.detach(),
                "anchor_logits": anchor_logits.detach(),
            }
            aux = {
                "mode": self.prediction_mode,
                "base_object_logits": base_object_logits,
                "anchor_logits": anchor_logits,
                "shape_residual_logits": residuals["shape_residual_logits"],
                "boundary_residual_logits": residuals["boundary_residual_logits"],
                "residual_logits": fusion_aux["residual_logits"],
                "final_object_logits": final_object_logits,
                "slot_weights": slot_weights,
                "slot_entropy": slot_aux["slot_entropy"],
                "slot_area": slot_aux["slot_area"],
                "slot_area_ed": slot_aux["slot_area_ed"],
                "slot_area_early_systole": slot_aux["slot_area_early_systole"],
                "slot_area_es": slot_aux["slot_area_es"],
                "slot_area_early_diastole": slot_aux["slot_area_early_diastole"],
                "slot_area_uncertain": slot_aux["slot_area_uncertain"],
                "slot_area_order_violation": slot_aux["slot_area_order_violation"],
                "slot_order_loss": slot_aux["slot_order_loss"],
                "ed_slot_usage": slot_aux["ed_slot_usage"],
                "es_slot_usage": slot_aux["es_slot_usage"],
                "phase_embed": phase_embed,
                "phase_descriptor": phase_descriptor,
                "phase_source": phase_source_code,
                "phase_reliability": phase_reliability,
                "phase_entropy": -(slot_weights * slot_weights.clamp_min(1.0e-8).log()).sum(dim=-1),
                "z_state": z,
                "z_delta": dz,
                "state_norm": z.detach().pow(2).mean(dim=-1).sqrt(),
                "state_delta_norm": dz.detach().pow(2).mean(dim=-1).sqrt(),
                "ode_update_norm": torch.tanh(dz.detach()).pow(2).mean(dim=-1).sqrt(),
                "anchor_features": anchor_features,
                "confidence": residuals["confidence"],
                "gate_mean_low": residuals["gate_low"].mean(),
                "gate_mean_mid": residuals["gate_mid"].mean(),
                "gate_mean_high": residuals["gate_high"].mean(),
                "inject_gate_low": residuals["gate_low"].mean(),
                "inject_gate_mid": residuals["gate_mid"].mean(),
                "inject_gate_high": residuals["gate_high"].mean(),
                "inject_gate_dec": residuals["anchor_trust"].mean(),
                "confidence_mean": residuals["confidence"].mean(),
                "confidence_std": residuals["confidence"].std(unbiased=False),
                "trust_mean": fusion_aux["trust_mean"],
                "trust_std": fusion_aux["trust_std"],
                "anchor_trust_ratio": fusion_aux["anchor_trust_ratio"],
                "image_trust_ratio": fusion_aux["image_trust_ratio"],
                "residual_abs_mean": fusion_aux["residual_abs_mean"],
                "residual_abs_max": fusion_aux["residual_abs_max"],
                "delta_abs_mean": fusion_aux["delta_abs_mean"],
                "residual_l1": fusion_aux["residual_logits"].abs().mean(),
                "residual_l2": fusion_aux["residual_logits"].pow(2).mean().sqrt(),
                "shape_residual_norm": residuals["shape_residual_logits"].abs().mean(),
                "boundary_residual_norm": residuals["boundary_residual_logits"].abs().mean(),
                "anchor_area": anchor_area,
            }
            out[f"memory_aux_{ti}"] = {"functional_anchor_aux": aux}
            should_detach = self.detach_state and self.detach_every != 0 and (
                self.detach_every <= 1 or (ti + 1) % self.detach_every == 0
            )
            prev_z = z.detach() if should_detach else z
            prev = {
                "z": prev_z,
                "area": anchor_area,
                "area_velocity": next_area_velocity.detach(),
                "slot_weights": slot_weights.detach(),
            }
        return out
