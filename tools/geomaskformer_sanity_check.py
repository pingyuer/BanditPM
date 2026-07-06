from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.registry import build_model


def _load_cfg(path: str):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(config_path.parent)):
        cfg = compose(config_name=config_path.name)
    OmegaConf.resolve(cfg)
    return cfg


def _count(module) -> int:
    return sum(param.numel() for param in module.parameters()) if module is not None else 0


def _parameter_accounting(model) -> dict[str, float]:
    image = _count(getattr(model, "image_tokenizer", None))
    mask = _count(getattr(model, "mask_tokenizer", None))
    prompt = _count(getattr(model, "prompt_query_adapter", None))
    transformer = _count(getattr(model, "transformer", None))
    pixel = _count(getattr(model, "pixel_decoder", None))
    proposal_module = getattr(model, "proposal_decoder", None)
    quality = _count(getattr(proposal_module, "quality", None))
    proposal = max(_count(proposal_module) - quality, 0)
    total = _count(model)
    unclassified = max(total - image - mask - prompt - transformer - pixel - proposal - quality, 0)
    return {
        "parameters_total": float(total),
        "parameter_accounting_error": float(
            abs(total - image - mask - prompt - transformer - pixel - proposal - quality - unclassified)
        ),
        "unclassified_parameter_count": float(unclassified),
    }


def _batch(batch_size: int = 2, frames: int = 3, size: int = 64, device: str = "cpu"):
    rgb = torch.randn(batch_size, frames, 1, size, size, device=device)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, device=device),
        torch.linspace(-1.0, 1.0, size, device=device),
        indexing="ij",
    )
    masks = []
    for ti in range(frames):
        radius = 0.35 + 0.04 * ti
        masks.append(((xx**2 + yy**2) < radius**2).float())
    cls_gt = torch.stack(masks, dim=0).reshape(1, frames, 1, size, size).repeat(batch_size, 1, 1, 1, 1)
    return {
        "rgb": rgb,
        "cls_gt": cls_gt.long(),
        "label_valid": torch.ones(batch_size, frames, dtype=torch.bool, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/geomaskformer_stage1_echo.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    model = build_model(cfg, device=args.device).to(args.device)
    model.eval()
    data = _batch(device=args.device)

    no_mask = {**data, "geomaskformer_mask_visibility": torch.zeros(2, 3, dtype=torch.long, device=args.device)}
    visible = {
        **data,
        "geomaskformer_mask_visibility": torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.long, device=args.device),
        "geomaskformer_loss_visibility": torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.bool, device=args.device),
    }
    with torch.no_grad():
        out_nomask_a = model(no_mask)
        out_nomask_b = model(no_mask)
        out_visible = model(visible)

    invariant = (out_nomask_a["logits"] - out_nomask_b["logits"]).abs().mean()
    sensitivity = (out_nomask_a["logits"] - out_visible["logits"]).abs().mean()
    visibility = visible["geomaskformer_mask_visibility"].bool()
    loss_visibility = visible["geomaskformer_loss_visibility"].bool()
    proposal_scores = out_visible["quality_scores"]
    report = {
        "mask_token_invariant_error": float(invariant.item()),
        "visible_condition_sensitivity": float(sensitivity.item()),
        "visibility_overlap_violation_count": float((visibility & loss_visibility).float().sum().item()),
        "proposal_score_nan_count": float(torch.isnan(proposal_scores).float().sum().item()),
        **_parameter_accounting(model),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    assert report["proposal_score_nan_count"] == 0.0
    assert report["visible_condition_sensitivity"] > 0.0
    assert report["parameter_accounting_error"] < 1.0e-6


if __name__ == "__main__":
    main()
