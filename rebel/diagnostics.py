from __future__ import annotations

import torch


def scalar(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().mean()


def summarize_rebel_aux(aux_items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    keys = [
        "r_obs",
        "write_fast",
        "write_slow",
        "decay_fast",
        "decay_slow",
        "disagreement",
        "memory_prior_area",
        "final_minus_base_abs",
        "final_minus_memory_abs",
        "belief_feature_delta_norm",
        "w_mask_delta",
        "s_mask_delta",
        "w_feat_delta",
        "s_feat_delta",
        "offset_obs_px",
        "offset_mem_px",
        "correction_scale",
        "corrected_minus_rebel_abs",
        "arbitration_entropy",
        "arbitration_temperature",
        "arbitration_weight_base",
        "arbitration_weight_obs",
        "arbitration_weight_belief",
        "arbitration_weight_rebel",
        "arbitration_weight_corrected",
    ]
    for key in keys:
        vals = [item[key].detach().float().reshape(-1) for item in aux_items if key in item and torch.is_tensor(item[key])]
        if vals:
            cat = torch.cat(vals)
            out[f"rebel/{key}_mean"] = cat.mean()
            if key in {"r_obs"}:
                out[f"rebel/{key}_std"] = cat.std(unbiased=False)
            if key in {"offset_obs_px", "offset_mem_px"}:
                out[f"rebel/{key}_max"] = cat.max()
    return out
