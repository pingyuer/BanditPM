from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from models.registry import build_model


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg(config_name: str, overrides: list[str]) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(_repo_root() / "config")):
        return compose(config_name=config_name, overrides=overrides)


def _synthetic_batch(batch_size: int, frames: int, size: int, device: torch.device, *, init_mode: str) -> dict:
    rgb = torch.rand(batch_size, frames, 1, size, size, device=device)
    masks = (torch.rand(batch_size, frames, 1, size, size, device=device) > 0.82).long()
    ff_gt = masks[:, :1].clone()
    return {
        "rgb": rgb,
        "ff_gt": ff_gt,
        "cls_gt": masks,
        "label_valid": torch.ones(batch_size, frames, device=device, dtype=torch.bool),
        "eval_valid": torch.ones(batch_size, frames, device=device, dtype=torch.bool),
        "selector": torch.ones(batch_size, 1, device=device, dtype=torch.float32),
        "info": {"num_objects": torch.ones(batch_size, device=device, dtype=torch.long)},
        "original_size": torch.full((batch_size, frames, 2), size, device=device, dtype=torch.long),
        "resized_size": torch.full((batch_size, frames, 2), size, device=device, dtype=torch.long),
        "frame_indices": torch.arange(frames, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1),
        "init_mode": init_mode,
        "current_iter": 0,
        "current_epoch": 0,
        "global_step": 0,
        "iters_per_epoch": 1,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_mb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return torch.cuda.memory_allocated(device) / 1024**2, torch.cuda.max_memory_allocated(device) / 1024**2


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def _benchmark_one(
    *,
    name: str,
    config_name: str,
    device: torch.device,
    batch_size: int,
    frames: int,
    size: int,
    warmup: int,
    iters: int,
    amp: bool,
    init_mode: str,
    overrides: list[str],
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    cfg = _load_cfg(config_name, overrides)
    model = build_model(cfg, device=device)
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    param_count = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    param_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    model_allocated_mb, _ = _memory_mb(device)

    batch = _synthetic_batch(batch_size, frames, size, device, init_mode=init_mode)
    timings_ms: list[float] = []
    autocast_enabled = amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated_mb, _ = _memory_mb(device)

    with torch.inference_mode():
        for _ in range(warmup):
            with torch.amp.autocast(device.type, enabled=autocast_enabled):
                _ = model(batch)
            _sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated_mb, _ = _memory_mb(device)
        for _ in range(iters):
            _sync(device)
            start = time.perf_counter()
            with torch.amp.autocast(device.type, enabled=autocast_enabled):
                _ = model(batch)
            _sync(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    allocated_mb, peak_mb = _memory_mb(device)
    del model, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mean_ms = statistics.mean(timings_ms) if timings_ms else 0.0
    median_ms = statistics.median(timings_ms) if timings_ms else 0.0
    p95_ms = _percentile(timings_ms, 0.95)
    sequences_per_s = 1000.0 * batch_size / mean_ms if mean_ms > 0 else 0.0
    frames_per_s = sequences_per_s * frames

    return {
        "name": name,
        "config": config_name,
        "device": str(device),
        "batch_size": batch_size,
        "frames": frames,
        "size": size,
        "amp": autocast_enabled,
        "parameters_m": param_count / 1e6,
        "trainable_parameters_m": trainable_count / 1e6,
        "parameter_memory_mb": param_bytes / 1024**2,
        "model_allocated_mb": model_allocated_mb,
        "forward_baseline_allocated_mb": baseline_allocated_mb,
        "forward_peak_allocated_mb": peak_mb,
        "forward_peak_delta_mb": max(0.0, peak_mb - baseline_allocated_mb),
        "final_allocated_mb": allocated_mb,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "sequences_per_s": sequences_per_s,
        "frames_per_s": frames_per_s,
        "warmup": warmup,
        "iters": iters,
        "init_mode": init_mode,
    }


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inference_benchmark.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "inference_benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "name",
        "params(M)",
        "model MB",
        "peak delta MB",
        "mean ms",
        "median ms",
        "p95 ms",
        "seq/s",
        "frames/s",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["name"]),
                    f"{row['parameters_m']:.2f}",
                    f"{row['model_allocated_mb']:.1f}",
                    f"{row['forward_peak_delta_mb']:.1f}",
                    f"{row['mean_ms']:.2f}",
                    f"{row['median_ms']:.2f}",
                    f"{row['p95_ms']:.2f}",
                    f"{row['sequences_per_s']:.2f}",
                    f"{row['frames_per_s']:.2f}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference memory and speed for GDKVM-family models.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for inference timing.")
    parser.add_argument("--init-mode", default="pred_or_zero", choices=["pred_or_zero", "oracle_gt"])
    parser.add_argument("--output-dir", default="outputs/benchmarks/inference")
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "CONFIG"),
        help="Model pair to benchmark. Can be repeated. Defaults to functional_anchor and gdkvm CAMUS configs.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides applied to every model config.")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    model_specs = args.model or [
        ["functional_anchor", "functional_anchor_camus.yaml"],
        ["gdkvm", "gdkvm_camus.yaml"],
    ]
    rows = []
    for name, config_name in model_specs:
        rows.append(
            _benchmark_one(
                name=name,
                config_name=config_name,
                device=device,
                batch_size=args.batch_size,
                frames=args.frames,
                size=args.size,
                warmup=args.warmup,
                iters=args.iters,
                amp=args.amp,
                init_mode=args.init_mode,
                overrides=args.overrides,
            )
        )
    _write_outputs(rows, Path(args.output_dir))
    _print_table(rows)
    print(f"Wrote: {Path(args.output_dir) / 'inference_benchmark.json'}")
    print(f"Wrote: {Path(args.output_dir) / 'inference_benchmark.csv'}")


if __name__ == "__main__":
    main()
