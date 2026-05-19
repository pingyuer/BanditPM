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
        panel_names.extend(["Base aux", "Anchor only", "Residual"])
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
    
    save_path = Path(run_path) / filename
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Successfully saved sequence visualization with {num_frames} frames to {save_path}")
    return save_path
