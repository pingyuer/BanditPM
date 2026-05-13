#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Export delay_ode slot/area/gate curves from a saved delay_ode_aux dict.")
    parser.add_argument("--aux-pt", required=True, help="Path to a torch-saved delay_ode_aux dict or full memory_aux dict.")
    parser.add_argument("--out-dir", required=True, help="Directory for CSV diagnostics.")
    args = parser.parse_args()

    aux_path = Path(args.aux_pt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(aux_path, map_location="cpu")
    aux = payload.get("delay_ode_aux", payload) if isinstance(payload, dict) else payload
    aux = _to_cpu(aux)

    weights = aux.get("keymap_weights", {})
    gates = aux.get("update_gates", {})
    stats = aux.get("mask_stats")

    for level, tensor in weights.items():
        path = out_dir / f"{level}_slot_weights.csv"
        # Expected [B, N, T, K]. Export sample/object 0 by default.
        values = tensor[0, 0] if tensor.numel() else torch.empty(0, 0)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame"] + [f"slot_{i}" for i in range(values.shape[-1] if values.ndim == 2 else 0)])
            for t, row in enumerate(values):
                writer.writerow([t + 1] + row.tolist())

    for level, tensor in gates.items():
        path = out_dir / f"{level}_update_gate.csv"
        values = tensor[0, 0, :, 0] if tensor.numel() else torch.empty(0)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "gate"])
            for t, value in enumerate(values):
                writer.writerow([t + 1, float(value)])

    if torch.is_tensor(stats) and stats.numel():
        names = ["area", "cx", "cy", "width", "height", "entropy", "area_v", "cx_v", "cy_v", "scale_v", "entropy_v"]
        values = stats[0, 0]
        with (out_dir / "mask_stats.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame"] + names[: values.shape[-1]])
            for t, row in enumerate(values):
                writer.writerow([t] + row.tolist())

    print(f"Wrote delay_ode diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
