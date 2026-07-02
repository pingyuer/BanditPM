from __future__ import annotations

import torch
import torch.nn as nn

from rebel.ode_field import ConvNeXtLiteBlock, _groups
from rebel.resampling import sample_feature


class ResampledBeliefMemory(nn.Module):
    def __init__(self, belief_dim: int, stable_reliability_threshold: float = 0.55, detach_update: bool = True) -> None:
        super().__init__()
        self.belief_dim = int(belief_dim)
        self.stable_reliability_threshold = float(stable_reliability_threshold)
        self.detach_update = bool(detach_update)
        hidden = belief_dim
        self.read = nn.Sequential(
            nn.Conv2d(belief_dim * 2 + 3, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(),
            ConvNeXtLiteBlock(hidden),
            ConvNeXtLiteBlock(hidden),
        )
        self.mem_proj = nn.Conv2d(hidden, belief_dim, 1)
        self.mask_prior_head = nn.Conv2d(belief_dim, 1, 1)
        self.arbiter = nn.Sequential(
            nn.Conv2d(belief_dim * 2 + 4, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, 1),
        )

    def init_state(self, obs: torch.Tensor, mask_prob: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b, _, h, w = obs.shape
        if mask_prob is None:
            mask_prob = obs.new_full((b, 1, h, w), 0.5)
        reliability = obs.new_full((b, 1, h, w), 0.5)
        return {
            "W_feat": obs.detach().clone(),
            "S_feat": obs.detach().clone(),
            "W_mask": mask_prob.detach().clone(),
            "S_mask": mask_prob.detach().clone(),
            "R": reliability,
        }

    def read_memory(self, state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state["W_feat"], state["S_feat"], state["W_mask"], state["S_mask"], state["R"]], dim=1)
        mem = self.mem_proj(self.read(x))
        return mem, self.mask_prior_head(mem)

    def resample_state(self, state: dict[str, torch.Tensor], delta_mem: torch.Tensor) -> dict[str, torch.Tensor]:
        return {key: sample_feature(value, delta_mem) for key, value in state.items()}

    def arbitrate(
        self,
        obs_belief: torch.Tensor,
        mem_belief: torch.Tensor,
        mem_prob: torch.Tensor,
        obs_prob: torch.Tensor,
        reliability: torch.Tensor,
        r_obs_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        disagreement = (obs_prob - mem_prob).abs()
        arb_in = torch.cat([obs_belief, mem_belief, mem_prob, obs_prob, disagreement, reliability], dim=1)
        obs_rel_raw, obs_gate_raw = self.arbiter(arb_in).chunk(2, dim=1)
        obs_rel = torch.sigmoid(obs_rel_raw)
        obs_gate = 0.5 * (torch.sigmoid(obs_gate_raw) + r_obs_raw.clamp(0.0, 1.0))
        mem_rel = reliability.clamp(0.0, 1.0)
        obs_weight = obs_rel * obs_gate
        mem_weight = mem_rel * (1.0 - obs_gate)
        r_obs = obs_weight / (obs_weight + mem_weight + 1.0e-6)
        belief = r_obs * obs_belief + (1.0 - r_obs) * mem_belief
        return belief, r_obs, disagreement

    def update(
        self,
        state_hat: dict[str, torch.Tensor],
        belief: torch.Tensor,
        final_prob: torch.Tensor,
        write_fast: torch.Tensor,
        write_slow: torch.Tensor,
        decay_fast: torch.Tensor,
        decay_slow: torch.Tensor,
        disagreement: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        src_belief = belief.detach() if self.detach_update else belief
        src_prob = final_prob.detach() if self.detach_update else final_prob
        stable_gate = (state_hat["R"] >= self.stable_reliability_threshold).to(write_slow.dtype)
        write_slow_eff = write_slow * stable_gate * state_hat["R"]
        w_feat = state_hat["W_feat"] + write_fast * (src_belief - state_hat["W_feat"])
        s_feat = state_hat["S_feat"] + write_slow_eff * (src_belief - state_hat["S_feat"])
        w_mask = state_hat["W_mask"] + write_fast * (src_prob - state_hat["W_mask"])
        s_mask = state_hat["S_mask"] + write_slow_eff * (src_prob - state_hat["S_mask"])
        w_feat = decay_fast * w_feat + (1.0 - decay_fast) * state_hat["W_feat"]
        s_feat = decay_slow * s_feat + (1.0 - decay_slow) * state_hat["S_feat"]
        w_mask = decay_fast * w_mask + (1.0 - decay_fast) * state_hat["W_mask"]
        s_mask = decay_slow * s_mask + (1.0 - decay_slow) * state_hat["S_mask"]
        next_r = (state_hat["R"] * (1.0 - 0.25 * disagreement)).clamp(0.0, 1.0)
        next_r = (next_r + write_fast * (1.0 - disagreement)).clamp(0.0, 1.0)
        return {"W_feat": w_feat, "S_feat": s_feat, "W_mask": w_mask, "S_mask": s_mask, "R": next_r}
