from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rebel.correction import DisagreementCorrectionHead
from rebel.decoder import BeliefDecoder
from rebel.diagnostics import summarize_rebel_aux
from rebel.encoder import ObservationEncoder
from rebel.fusion import BeliefLogitArbitration
from rebel.memory import ResampledBeliefMemory
from rebel.ode_field import BeliefODEField
from rebel.resampling import sample_feature


def _cfg_get(cfg, key: str, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else default


class ReBelSegmenter(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        rebel_cfg = _cfg_get(cfg, "rebel", cfg)
        self.num_classes = int(_cfg_get(rebel_cfg, "num_classes", _cfg_get(cfg, "num_classes", 2)))
        self.belief_dim = int(_cfg_get(rebel_cfg, "belief_dim", 256))
        self.decoder_dim = int(_cfg_get(rebel_cfg, "decoder_dim", 192))
        self.encoder = ObservationEncoder(rebel_cfg)
        ode_cfg = _cfg_get(rebel_cfg, "ode", {})
        mem_cfg = _cfg_get(rebel_cfg, "memory", {})
        corr_cfg = _cfg_get(rebel_cfg, "correction", {})
        fusion_cfg = _cfg_get(rebel_cfg, "fusion", {})
        self.correction_start_iter = int(_cfg_get(corr_cfg, "start_iter", 1000))
        self.ode = BeliefODEField(
            self.belief_dim,
            hidden_dim=int(_cfg_get(ode_cfg, "hidden_dim", self.decoder_dim)),
            num_blocks=int(_cfg_get(ode_cfg, "num_blocks", 3)),
            max_offset_px=float(_cfg_get(ode_cfg, "max_offset_px_stage4", 6.0)),
            offset_warmup_iters=int(_cfg_get(ode_cfg, "offset_warmup_iters", 800)),
            offset_warmup_start_ratio=float(_cfg_get(ode_cfg, "offset_warmup_start_ratio", 0.25)),
            write_fast_init=float(_cfg_get(mem_cfg, "write_fast_init", 0.25)),
            write_slow_init=float(_cfg_get(mem_cfg, "write_slow_init", 0.05)),
            decay_fast_init=float(_cfg_get(mem_cfg, "decay_fast_init", 0.75)),
            decay_slow_init=float(_cfg_get(mem_cfg, "decay_slow_init", 0.95)),
        )
        self.memory = ResampledBeliefMemory(
            self.belief_dim,
            stable_reliability_threshold=float(_cfg_get(mem_cfg, "stable_reliability_threshold", 0.55)),
            detach_update=bool(_cfg_get(mem_cfg, "detach_update", True)),
        )
        self.obs_head = nn.Conv2d(self.belief_dim, self.num_classes, 1)
        self.decoder = BeliefDecoder(
            self.belief_dim,
            low_dim=self.encoder.low_dim,
            mid_dim=self.encoder.mid_dim,
            high_dim=self.encoder.high_dim,
            decoder_dim=self.decoder_dim,
            num_classes=self.num_classes,
        )
        self.correction_enabled = bool(_cfg_get(corr_cfg, "enabled", True))
        self.correction = DisagreementCorrectionHead(
            self.decoder_dim,
            num_classes=self.num_classes,
            init_scale=float(_cfg_get(corr_cfg, "init_scale", 0.25)),
            max_scale=float(_cfg_get(corr_cfg, "max_scale", 1.0)),
        )
        self.fusion = BeliefLogitArbitration(
            self.decoder_dim,
            num_classes=self.num_classes,
            hidden_dim=int(_cfg_get(fusion_cfg, "hidden_dim", self.decoder_dim)),
            init_base_bias=float(_cfg_get(fusion_cfg, "init_base_bias", 1.2)),
            init_obs_bias=float(_cfg_get(fusion_cfg, "init_obs_bias", 0.4)),
            init_belief_bias=float(_cfg_get(fusion_cfg, "init_belief_bias", -0.2)),
            init_rebel_bias=float(_cfg_get(fusion_cfg, "init_rebel_bias", 0.0)),
            init_corrected_bias=float(_cfg_get(fusion_cfg, "init_corrected_bias", -0.4)),
            init_temperature=float(_cfg_get(fusion_cfg, "init_temperature", 1.25)),
            min_base_weight=float(_cfg_get(fusion_cfg, "min_base_weight", 0.05)),
        )

    def _foreground_prob(self, logits: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        if size is not None and logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        if logits.shape[1] == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)[:, 1:2]

    def _belief_logits_from_prior(self, prior_logits: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        fg = F.interpolate(prior_logits, size=output_size, mode="bilinear", align_corners=False)
        return torch.cat([torch.zeros_like(fg), fg], dim=1)

    def forward(self, data: Dict) -> Dict:
        images = data["rgb"]
        if images.dim() != 5:
            raise ValueError("ReBel expects batch['rgb'] with shape B,T,C,H,W.")
        b, t, _, h, w = images.shape
        current_iter = int(data.get("current_iter", 0)) if isinstance(data, dict) else 0
        out: Dict[str, torch.Tensor | dict] = {}
        state = None
        stacks = {"logits": [], "base_logits": [], "belief_logits": [], "obs_logits": [], "correction_logits": []}
        aux_items = []
        for ti in range(t):
            feats = self.encoder(images[:, ti])
            obs = feats["obs"]
            base_logits = feats["base_logits"]
            base_prob_low = self._foreground_prob(base_logits, obs.shape[-2:])
            if state is None:
                state = self.memory.init_state(obs, base_prob_low)
            mem_proxy, mem_prior_logits_proxy = self.memory.read_memory(state)
            field = self.ode(obs, state["W_feat"], state["S_feat"], torch.sigmoid(mem_prior_logits_proxy), state["R"], current_iter)
            obs_belief = sample_feature(obs, field["delta_obs"])
            state_hat = self.memory.resample_state(state, field["delta_mem"])
            mem_belief, mem_prior_logits = self.memory.read_memory(state_hat)
            obs_logits_low = self.obs_head(obs_belief)
            obs_prob = self._foreground_prob(obs_logits_low)
            mem_prob = torch.sigmoid(mem_prior_logits)
            belief, r_obs, disagreement = self.memory.arbitrate(
                obs_belief, mem_belief, mem_prob, obs_prob, state_hat["R"], field["r_obs"]
            )
            rebel_logits, rebel_feature = self.decoder(
                belief,
                {"low": feats["low"], "mid": feats["mid"], "high": feats["high"]},
                disagreement,
                state_hat["R"],
                (h, w),
            )
            belief_logits = self._belief_logits_from_prior(mem_prior_logits, (h, w))
            correction_logits = torch.zeros_like(rebel_logits)
            correction_scale = rebel_logits.new_tensor(0.0)
            if self.correction_enabled and current_iter >= self.correction_start_iter:
                correction_logits, correction_scale = self.correction(
                    rebel_feature, obs_logits_low, belief_logits, disagreement, state_hat["R"]
                )
            corrected_logits = rebel_logits + correction_scale * correction_logits
            obs_logits_full = F.interpolate(obs_logits_low, size=(h, w), mode="bilinear", align_corners=False)
            final_logits, fusion_aux = self.fusion(
                base_logits=base_logits,
                obs_logits=obs_logits_full,
                belief_logits=belief_logits,
                rebel_logits=rebel_logits,
                corrected_logits=corrected_logits,
                rebel_feature=rebel_feature,
                disagreement=disagreement,
                reliability=state_hat["R"],
            )
            final_prob_low = self._foreground_prob(final_logits, obs.shape[-2:])
            next_state = self.memory.update(
                state_hat,
                belief,
                final_prob_low,
                field["write_fast"],
                field["write_slow"],
                field["decay_fast"],
                field["decay_slow"],
                disagreement,
            )
            w_feat_delta = (next_state["W_feat"] - state_hat["W_feat"]).abs().mean(dim=1, keepdim=True)
            s_feat_delta = (next_state["S_feat"] - state_hat["S_feat"]).abs().mean(dim=1, keepdim=True)
            w_mask_delta = (next_state["W_mask"] - state_hat["W_mask"]).abs()
            s_mask_delta = (next_state["S_mask"] - state_hat["S_mask"]).abs()
            state = next_state
            masks = torch.softmax(final_logits, dim=1)[:, 1:]
            aux = {
                "base_logits": base_logits,
                "belief_logits": belief_logits,
                "obs_logits": obs_logits_full,
                "rebel_logits": rebel_logits,
                "corrected_logits": corrected_logits,
                "correction_logits": correction_logits,
                "rebel_feature": rebel_feature,
                "rebel/disagreement": disagreement,
                "rebel/r_obs": r_obs,
            }
            aux.update(fusion_aux)
            frame_aux = {
                "r_obs": r_obs,
                "write_fast": field["write_fast"],
                "write_slow": field["write_slow"],
                "decay_fast": field["decay_fast"],
                "decay_slow": field["decay_slow"],
                "disagreement": disagreement,
                "memory_prior_area": mem_prob,
                "final_minus_base_abs": (final_logits - base_logits).abs(),
                "final_minus_memory_abs": (final_logits - belief_logits).abs(),
                "corrected_minus_rebel_abs": (corrected_logits - rebel_logits).abs(),
                "belief_feature_delta_norm": (belief - state_hat["W_feat"]).pow(2).mean(dim=1, keepdim=True).sqrt(),
                "w_feat_delta": w_feat_delta,
                "s_feat_delta": s_feat_delta,
                "w_mask_delta": w_mask_delta,
                "s_mask_delta": s_mask_delta,
                "offset_obs_px": field["delta_obs"].abs(),
                "offset_mem_px": field["delta_mem"].abs(),
                "correction_scale": correction_scale.reshape(1),
                "arbitration_entropy": fusion_aux["arbitration_entropy"],
                "arbitration_temperature": fusion_aux["arbitration_temperature"],
            }
            for name in self.fusion.candidate_names:
                frame_aux[f"arbitration_weight_{name}"] = fusion_aux[f"arbitration_weight_{name}"]
            aux_items.append(frame_aux)
            out[f"logits_{ti}"] = final_logits
            out[f"masks_{ti}"] = masks
            out[f"aux_{ti}"] = aux
            out[f"memory_aux_{ti}"] = {"rebel_aux": frame_aux}
            stacks["logits"].append(final_logits)
            stacks["base_logits"].append(base_logits)
            stacks["belief_logits"].append(belief_logits)
            stacks["obs_logits"].append(obs_logits_full)
            stacks["correction_logits"].append(correction_logits)
        for key, frames in stacks.items():
            out[key] = torch.stack(frames, dim=1)
        out["aux"] = summarize_rebel_aux(aux_items)
        return out
