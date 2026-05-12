#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

import torch
from omegaconf import OmegaConf

from model.unext_dynakey import UNeXtDynaKeySegmenter


def _base_cfg(*, spatial: bool, q_mode: str = "off"):
    return OmegaConf.create(
        {
            "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
            "memory_core": {
                "type": "dynakey",
                "dynakey": {"BANK_SIZE": 3, "HIDDEN_DIM": 16, "POLICY_MODE": "fixed_residual", "ENABLE_Q_LOSS": False},
            },
            "temporal_memory": {"type": "dynakey", "bpm": {}},
            "allow_oracle_init_when_requested": False,
            "unext_dynakey": {
                "in_channels": 1,
                "num_classes": 2,
                "base_dim": 8,
                "value_dim": 16,
                "use_dynakey": spatial,
                "dynakey_memory_mode": "spatial" if spatial else "global",
                "use_phase_retrieval": spatial,
                "readout_type": "spatial_gate" if spatial else "global_broadcast",
                "dynamics_mode": "spatial" if spatial else "none",
                "q_policy_mode": q_mode,
                "enable_q_loss": q_mode == "training",
                "use_temporal_refine": True,
                "use_mask_memory": False,
                "use_memory_readout": False,
                "spatial_memory_slots": 3,
                "spatial_memory_size": 8,
                "spatial_memory_confidence_threshold": 0.0,
                "spatial_memory_fg_ratio_min": 0.0,
                "spatial_memory_fg_ratio_max": 1.0,
                "temporal_residual_init_scale": 0.1,
                "temporal_gate_bias": -2.0,
            },
        }
    )


def _batch(batch_size: int, frames: int, height: int, width: int):
    rgb = torch.randn(batch_size, frames, 1, height, width)
    cls_gt = torch.zeros(batch_size, frames, 1, height, width, dtype=torch.long)
    for t in range(frames):
        y0 = height // 4 + t
        x0 = width // 4 + t
        cls_gt[:, t, :, y0 : y0 + height // 4, x0 : x0 + width // 4] = 1
    ff_gt = torch.zeros(batch_size, 1, 1, height, width)
    return {
        "rgb": rgb,
        "ff_gt": ff_gt,
        "cls_gt": cls_gt,
        "label_valid": torch.ones(batch_size, frames, dtype=torch.bool),
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
        "init_mode": "pred_or_zero",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic spatial-phase DynaKey debug probe.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--q-mode", choices=("off", "diagnostic", "training"), default="diagnostic")
    args = parser.parse_args()

    torch.manual_seed(7)
    data = _batch(args.batch_size, args.frames, args.height, args.width)
    refine_only = UNeXtDynaKeySegmenter(_base_cfg(spatial=False)).eval()
    spatial = UNeXtDynaKeySegmenter(_base_cfg(spatial=True, q_mode=args.q_mode)).eval()
    with torch.no_grad():
        out_refine = refine_only(data)
        out_spatial = spatial(data)

    diffs = []
    gate_means = []
    gate_stds = []
    gate_maxes = []
    delta_norms = []
    entropies = []
    slots = []
    q_entropies = []
    areas = []
    area_deltas = []
    for t in range(args.frames):
        diffs.append((out_spatial[f"logits_{t}"] - out_refine[f"logits_{t}"]).abs().mean().item())
        aux = out_spatial.get(f"memory_aux_{t}", {})
        for key, target in (
            ("spatial_gate_mean", gate_means),
            ("spatial_gate_std", gate_stds),
            ("spatial_gate_max", gate_maxes),
        ):
            value = aux.get(key)
            if torch.is_tensor(value):
                target.append(float(value.detach().mean()))
        value = aux.get("spatial_delta_norm")
        if torch.is_tensor(value):
            delta_norms.append(float(value.detach().mean()))
        value = aux.get("spatial_memory_entropy")
        if torch.is_tensor(value):
            entropies.append(float(value.detach().mean()))
        value = aux.get("spatial_memory_top_slot")
        if torch.is_tensor(value):
            slots.extend(int(x) for x in value.detach().flatten().tolist())
        value = aux.get("spatial_q_entropy")
        if torch.is_tensor(value):
            q_entropies.append(float(value.detach().mean()))
        value = aux.get("phase_area")
        if torch.is_tensor(value):
            areas.append(float(value.detach().mean()))
        value = aux.get("phase_area_delta")
        if torch.is_tensor(value):
            area_deltas.append(float(value.detach().mean()))

    print(f"logits_abs_diff_mean={sum(diffs) / max(len(diffs), 1):.6f}")
    print(f"spatial_gate_mean={sum(gate_means) / max(len(gate_means), 1):.6f}")
    print(f"spatial_gate_std={sum(gate_stds) / max(len(gate_stds), 1):.6f}")
    print(f"spatial_gate_max={max(gate_maxes) if gate_maxes else 0.0:.6f}")
    print(f"spatial_delta_norm={sum(delta_norms) / max(len(delta_norms), 1):.6f}")
    print(f"retrieval_entropy={sum(entropies) / max(len(entropies), 1):.6f}")
    print(f"selected_slot_hist={dict(Counter(slots))}")
    print(f"area_curve={[round(x, 6) for x in areas]}")
    print(f"area_delta_curve={[round(x, 6) for x in area_deltas]}")
    if q_entropies:
        print(f"q_action_entropy={sum(q_entropies) / len(q_entropies):.6f}")


if __name__ == "__main__":
    main()
