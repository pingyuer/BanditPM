import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from pathlib import Path

def visualize_sequence(rgb_seq, cls_gt_seq, out_dict, run_path, batch_idx_str, iteration=None, epoch=None, patient_id=None, channel_type=None, mode='val'):
    logits_keys = sorted(
        [k for k in out_dict.keys() if k.startswith('logits_')],
        key=lambda x: int(x.split('_')[-1])
    )

    num_frames = len(logits_keys)
    if num_frames == 0:
        print("No frames to visualize.")
        return None

    functional_aux_by_t = []
    for key in logits_keys:
        ti = int(key.split('_')[-1])
        memory_aux = out_dict.get(f"memory_aux_{ti}", {})
        aux = memory_aux.get("functional_anchor_aux") if isinstance(memory_aux, dict) else None
        functional_aux_by_t.append(aux if isinstance(aux, dict) else None)
    has_functional_anchor = any(aux is not None for aux in functional_aux_by_t)
    panel_names = ["Original", "Overlay", "Final heatmap"]
    if has_functional_anchor:
        panel_names.extend(["Base aux", "Anchor only", "Residual", "Trust", "Slot/area"])
    panels_per_frame = len(panel_names)

    if num_frames > 15:
        frames_per_row = min(15, num_frames)
        num_rows_per_type = (num_frames + frames_per_row - 1) // frames_per_row
        total_rows = num_rows_per_type * panels_per_frame
        
        fig, axs = plt.subplots(total_rows, frames_per_row, figsize=(2.5 * frames_per_row, 3 * total_rows), squeeze=False)
    else:
        fig, axs = plt.subplots(panels_per_frame, num_frames, figsize=(4 * num_frames, 4 * panels_per_frame), squeeze=False)

    norm = Normalize(vmin=0, vmax=1)
    cmap_gt = ListedColormap([(0, 1, 0, 0.6)])
    cmap_pred = ListedColormap([(1, 0, 0, 0.6)])
    cmap_overlap = ListedColormap([(1, 1, 0, 0.8)])

    for t in range(num_frames):
        key = logits_keys[t]
        rgb_frame = rgb_seq[t, 0]
        gt_frame = cls_gt_seq[t, 0]
        logits_frame = out_dict[key][0]
        
        if logits_frame.shape[0] == 2:
            logits_frame = logits_frame[1:2, :, :]
            
        prob_frame = torch.sigmoid(logits_frame).detach().cpu().numpy().squeeze()
        pred_frame = (prob_frame > 0.5).astype(np.uint8)
        overlap_frame = np.logical_and(gt_frame, pred_frame)

        aux = functional_aux_by_t[t]

        # Determine axes
        if num_frames > 15:
            frame_row = t // frames_per_row
            frame_col = t % frames_per_row
            axes = [axs[frame_row + row_offset * num_rows_per_type, frame_col] for row_offset in range(panels_per_frame)]
        else:
            axes = [axs[row, t] for row in range(panels_per_frame)]
        ax_orig, ax_overlay, ax_heatmap = axes[:3]

        # Plot Original
        ax_orig.imshow(rgb_frame, cmap='gray')
        ax_orig.set_title(f"Frame {t}", fontsize=10 if num_frames > 15 else 12)
        ax_orig.axis('off')

        # Plot Overlay
        ax_overlay.imshow(rgb_frame, cmap='gray')
        ax_overlay.imshow(np.ma.masked_where(gt_frame == 0, np.ones_like(gt_frame)), cmap=cmap_gt)
        ax_overlay.imshow(np.ma.masked_where(pred_frame == 0, np.ones_like(pred_frame)), cmap=cmap_pred)
        ax_overlay.imshow(np.ma.masked_where(overlap_frame == 0, np.ones_like(overlap_frame)), cmap=cmap_overlap)
        ax_overlay.axis('off')

        # Plot Heatmap
        ax_heatmap.imshow(rgb_frame, cmap='gray')
        ax_heatmap.imshow(prob_frame, cmap='jet', alpha=0.5, interpolation='nearest', norm=norm)
        ax_heatmap.axis('off')

        if has_functional_anchor:
            for extra_ax in axes[3:]:
                extra_ax.imshow(rgb_frame, cmap='gray')
                extra_ax.axis('off')
            if aux is not None:
                base_logits = aux.get("base_object_logits")
                anchor_logits = aux.get("anchor_logits")
                residual_logits = aux.get("residual_logits")
                confidence = aux.get("confidence")
                slot_weights = aux.get("slot_weights")
                slot_area = aux.get("slot_area")
                if torch.is_tensor(base_logits):
                    base_prob = torch.sigmoid(base_logits[0, 0]).detach().cpu().numpy()
                    axes[3].imshow(base_prob, cmap='magma', alpha=0.55, interpolation='nearest', norm=norm)
                if torch.is_tensor(anchor_logits):
                    anchor_prob = torch.sigmoid(anchor_logits[0, 0]).detach().cpu().numpy()
                    axes[4].imshow(anchor_prob, cmap='viridis', alpha=0.55, interpolation='nearest', norm=norm)
                if torch.is_tensor(residual_logits):
                    residual = residual_logits[0, 0].detach().cpu().numpy()
                    vmax = max(float(np.abs(residual).max()), 1.0e-6)
                    axes[5].imshow(residual, cmap='coolwarm', alpha=0.65, interpolation='nearest', vmin=-vmax, vmax=vmax)
                if torch.is_tensor(confidence):
                    trust = confidence[0, 3].detach().cpu().numpy()
                    axes[6].imshow(trust, cmap='plasma', alpha=0.65, interpolation='nearest', norm=norm)
                if torch.is_tensor(slot_weights):
                    axes[7].clear()
                    weights = slot_weights[0, 0].detach().cpu().numpy()
                    x = np.arange(len(weights))
                    axes[7].bar(x, weights, color='tab:blue', alpha=0.75)
                    if torch.is_tensor(slot_area):
                        areas = slot_area.detach().cpu().numpy()
                        axes[7].plot(x, areas[: len(weights)], color='tab:red', marker='o', linewidth=1)
                    axes[7].set_ylim(0, 1)
                    axes[7].set_xticks([])

    if num_frames > 15:
        for i in range(num_rows_per_type):
            if i * frames_per_row < num_frames:
                for panel_idx, panel_name in enumerate(panel_names):
                    row_idx = i + panel_idx * num_rows_per_type
                    if row_idx < total_rows:
                        axs[row_idx, 0].set_ylabel(panel_name, fontsize=14, rotation=90, labelpad=20)
        
        for row in range(total_rows):
            for col in range(frames_per_row):
                frame_idx = (row % num_rows_per_type) * frames_per_row + col
                if frame_idx >= num_frames:
                    axs[row, col].axis('off')
    else:
        for panel_idx, panel_name in enumerate(panel_names):
            axs[panel_idx, 0].set_ylabel(panel_name, fontsize=16, rotation=90, labelpad=20)
    
    fig.subplots_adjust(wspace=0.02, hspace=0.005, top=0.95, bottom=0.05, left=0.08, right=0.95)
    
    filename_parts = []
    
    if iteration is not None:
        filename_parts.append(f"It_{iteration:04d}")
    
    if epoch is not None:
        filename_parts.append(f"E_{epoch:02d}")
    
    filename_parts.append(mode)
    
    if patient_id is not None:
        filename_parts.append(str(patient_id))
    
    filename = "_".join(filename_parts) + ".png"
    
    artifact_dir = Path(run_path) / "visuals"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_path = artifact_dir / filename
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    artifacts = [save_path]

    if has_functional_anchor:
        diag_paths = _save_functional_anchor_diagnostics(
            functional_aux_by_t,
            artifact_dir,
            filename.replace(".png", ""),
        )
        artifacts.extend(diag_paths)

    print(f"Successfully saved sequence visualization with {num_frames} frames to {save_path}")
    return artifacts


def _tensor_scalar(value, default=np.nan):
    if not torch.is_tensor(value):
        return default
    tensor = value.detach().float()
    if tensor.numel() == 0:
        return default
    return float(tensor.mean().cpu().item())


def _save_functional_anchor_diagnostics(functional_aux_by_t, artifact_dir: Path, stem: str):
    rows = []
    slot_rows = []
    slot_names = ["ed", "early_systole", "es", "early_diastole", "uncertain"]
    for ti, aux in enumerate(functional_aux_by_t):
        if aux is None:
            continue
        row = {
            "frame": ti,
            "base_area": _prob_area(aux.get("base_object_logits")),
            "anchor_area": _prob_area(aux.get("anchor_logits")),
            "proposal_area": _prob_area(aux.get("proposal_logits")),
            "final_area": _prob_area(aux.get("final_object_logits")),
            "residual_abs_mean": _tensor_scalar(aux.get("residual_abs_mean")),
            "trust_mean": _tensor_scalar(aux.get("trust_mean")),
            "phase_reliability": _tensor_scalar(aux.get("phase_reliability")),
            "delta_abs_mean": _tensor_scalar(aux.get("delta_abs_mean")),
        }
        rows.append(row)
        weights = aux.get("slot_weights")
        if torch.is_tensor(weights):
            values = weights[0, 0].detach().float().cpu().numpy().tolist()
            slot_row = {"frame": ti}
            for idx, value in enumerate(values):
                name = slot_names[idx] if idx < len(slot_names) else f"slot_{idx}"
                slot_row[name] = float(value)
            slot_rows.append(slot_row)

    paths = []
    if rows:
        csv_path = artifact_dir / f"{stem}_functional_anchor_curves.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(csv_path)

        curve_path = artifact_dir / f"{stem}_area_trust_residual.png"
        frames = [row["frame"] for row in rows]
        fig, axes = plt.subplots(3, 1, figsize=(max(6, len(frames) * 0.8), 8), squeeze=False)
        ax = axes[0, 0]
        for key in ("base_area", "anchor_area", "proposal_area", "final_area"):
            ax.plot(frames, [row[key] for row in rows], marker="o", label=key)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_ylabel("area")
        axes[1, 0].plot(frames, [row["trust_mean"] for row in rows], marker="o", color="tab:purple")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_ylabel("trust")
        axes[2, 0].plot(frames, [row["residual_abs_mean"] for row in rows], marker="o", color="tab:red")
        axes[2, 0].set_ylabel("residual")
        axes[2, 0].set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(curve_path)

    if slot_rows:
        slot_csv = artifact_dir / f"{stem}_slot_weights.csv"
        fieldnames = sorted({key for row in slot_rows for key in row.keys()}, key=lambda key: (key != "frame", key))
        with slot_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(slot_rows)
        paths.append(slot_csv)

        slot_png = artifact_dir / f"{stem}_slot_weights.png"
        frames = [row["frame"] for row in slot_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(frames) * 0.8), 3))
        for name in fieldnames:
            if name == "frame":
                continue
            ax.plot(frames, [row.get(name, np.nan) for row in slot_rows], marker="o", label=name)
        ax.set_ylim(0, 1)
        ax.set_xlabel("frame")
        ax.set_ylabel("slot weight")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(slot_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(slot_png)

    return paths


def _prob_area(logits):
    if not torch.is_tensor(logits):
        return np.nan
    tensor = torch.sigmoid(logits.detach().float())
    if tensor.dim() >= 4:
        tensor = tensor[0, 0]
    return float(tensor.mean().cpu().item())
