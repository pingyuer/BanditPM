from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import torch


def _to_numpy_image(tensor: torch.Tensor):
    x = tensor.detach().float().cpu()
    if x.dim() == 3:
        x = x[0]
    return x.clamp(0, 1).numpy()


def render_sequence_panel(
    batch: Mapping,
    output: Mapping,
    *,
    sample_idx: int = 0,
    max_frames: int = 4,
    save_path: str | Path | None = None,
):
    """Render image/ground-truth/prediction rows for a video sample."""
    rgb = batch["rgb"][sample_idx]
    gt = batch.get("cls_gt")
    logits = output.get("logits")
    frames = min(int(max_frames), int(rgb.shape[0]))
    fig, axes = plt.subplots(3, frames, figsize=(3 * frames, 7), squeeze=False)
    for ti in range(frames):
        axes[0, ti].imshow(_to_numpy_image(rgb[ti]), cmap="gray")
        axes[0, ti].set_title(f"frame {ti}")
        if torch.is_tensor(gt):
            axes[1, ti].imshow(gt[sample_idx, ti, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, ti].set_title("target")
        if torch.is_tensor(logits):
            pred = torch.softmax(logits[sample_idx, ti], dim=0)[1]
            axes[2, ti].imshow(pred.detach().cpu().numpy(), cmap="magma", vmin=0, vmax=1)
        axes[2, ti].set_title("prediction")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=140)
    return fig


def render_dpfr_diagnostic_panel(
    batch: Mapping,
    output: Mapping,
    *,
    sample_idx: int = 0,
    max_frames: int = 4,
    save_path: str | Path | None = None,
):
    """Render DPFR final/anchor/prompt/flow/error diagnostics."""
    rgb = batch["rgb"][sample_idx]
    gt = batch.get("cls_gt")
    if torch.is_tensor(gt):
        gt = gt[sample_idx]
        if gt.dim() == 4:
            gt = gt[:, 0]
    frames = min(int(max_frames), int(rgb.shape[0]))
    rows = [
        ("image", None),
        ("target", gt),
        ("final", output.get("final_logits", output.get("logits"))),
        ("anchor", output.get("anchor_logits")),
        ("prompt", output.get("prompt_logits")),
        ("flow", output.get("flow_logits")),
        ("error", None),
        ("flow_mag", output.get("flow_grid")),
    ]
    fig, axes = plt.subplots(len(rows), frames, figsize=(3 * frames, 2.3 * len(rows)), squeeze=False)
    for ti in range(frames):
        for row_idx, (name, value) in enumerate(rows):
            ax = axes[row_idx, ti]
            if name == "image":
                ax.imshow(_to_numpy_image(rgb[ti]), cmap="gray")
            elif name == "target" and torch.is_tensor(value):
                ax.imshow(value[ti].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            elif name == "error" and torch.is_tensor(gt) and torch.is_tensor(output.get("logits")):
                pred = (torch.softmax(output["logits"][sample_idx, ti].float(), dim=0)[1] >= 0.5).detach().cpu()
                err = pred.to(torch.int8) - gt[ti].detach().cpu().to(torch.int8)
                ax.imshow(err.numpy(), cmap="coolwarm", vmin=-1, vmax=1)
            elif name == "flow_mag" and torch.is_tensor(value):
                mag = value[sample_idx, ti].detach().float().pow(2).sum(dim=0).sqrt().cpu().numpy()
                ax.imshow(mag, cmap="viridis")
            elif torch.is_tensor(value):
                prob = torch.softmax(value[sample_idx, ti].float(), dim=0)[1]
                ax.imshow(prob.detach().cpu().numpy(), cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"{name} {ti}")
            ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=140)
    return fig
