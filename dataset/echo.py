import os
import json
import logging
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from dataset.frame_index import build_label_map


LOGGER = logging.getLogger(__name__)


def _infer_protocol_name(filepath: str) -> str:
    lower = filepath.lower()
    if "cardiacuda" in lower:
        if "dense" in lower:
            return "cardiacuda_a4c_lv_dense"
        return "cardiacuda_a4c_lv_sparse"
    if "full_cycle" in lower:
        return "echonet_fullcycle_sparse"
    return "echonet_ed2es_endpoint"

def _apply_intensity_augmentation(frames_t: torch.Tensor, augmentation_cfg) -> torch.Tensor:
    if not augmentation_cfg:
        return frames_t
    get = augmentation_cfg.get if hasattr(augmentation_cfg, "get") else lambda key, default=None: default
    if not bool(get("enabled", False)):
        return frames_t
    brightness = float(get("brightness", 0.0))
    contrast = float(get("contrast", 0.0))
    gamma = float(get("gamma", 0.0))
    if contrast > 0.0:
        scale = 1.0 + float(torch.empty((), device=frames_t.device).uniform_(-contrast, contrast).item())
        frames_t = (frames_t - 0.5) * scale + 0.5
    if brightness > 0.0:
        shift = float(torch.empty((), device=frames_t.device).uniform_(-brightness, brightness).item())
        frames_t = frames_t + shift
    frames_t = frames_t.clamp(0.0, 1.0)
    if gamma > 0.0:
        exponent = 1.0 + float(torch.empty((), device=frames_t.device).uniform_(-gamma, gamma).item())
        frames_t = frames_t.clamp_min(1.0e-6).pow(exponent)
    return frames_t.clamp(0.0, 1.0)


class EchoDataset(Dataset):
    """EchoNet-style sparse video dataset with keyframe supervision."""

    def __init__(
        self,
        filepath: str,
        mode: str = 'train',
        seq_length=10,
        max_num_obj=1,
        size=128,
        merge_probability=0.0,
        augmentation=None,
    ):
        super().__init__()
        self.filepath = filepath
        self.mode = mode
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size
        self.merge_probability = merge_probability
        self.augmentation = augmentation

        self.img_root = os.path.join(filepath, mode, 'img')
        self.label_root = os.path.join(filepath, mode, 'label')
        
        self.samples = []
        self.stats = {
            'total_video_count': 0,
            'accepted_video_count': 0,
            'skipped_short_video_count': 0,
            'skipped_no_label_count': 0,
            'skipped_label_mapping_failed_count': 0,
            'filtered_due_to_exact_length_old_rule_count': 0,
        }
        
        if os.path.isdir(self.img_root) and os.path.isdir(self.label_root):
            subfolders = sorted(os.listdir(self.img_root))
            
            for subfolder in subfolders:
                img_folder = os.path.join(self.img_root, subfolder)
                label_folder = os.path.join(self.label_root, subfolder)
                
                if os.path.isdir(img_folder) and os.path.isdir(label_folder):
                    self.stats['total_video_count'] += 1
                    img_files = sorted(os.listdir(img_folder))
                    label_files = sorted(os.listdir(label_folder))

                    if len(img_files) < self.seq_length:
                        self.stats['skipped_short_video_count'] += 1
                        continue
                    if not label_files:
                        self.stats['skipped_no_label_count'] += 1
                        continue
                    if len(img_files) != self.seq_length:
                        self.stats['filtered_due_to_exact_length_old_rule_count'] += 1
                    self.samples.append({
                        'subfolder': subfolder,
                        'img_folder': img_folder,
                        'label_folder': label_folder,
                        'img_files': img_files,
                        'label_files': label_files,
                        'meta_path': os.path.join(filepath, mode, 'metadata', f'{subfolder}.json'),
                    })
                    self.stats['accepted_video_count'] += 1
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_folder = sample['img_folder']
        label_folder = sample['label_folder']
        img_files = sample['img_files']
        label_files = sample['label_files']
        sample_meta = {}
        meta_path = sample.get('meta_path')
        if meta_path and os.path.isfile(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as handle:
                sample_meta = json.load(handle)

        imgs_np = np.zeros((self.seq_length, self.size, self.size), dtype=np.uint8)
        masks_np = np.zeros((self.seq_length, self.size, self.size), dtype=np.uint8)

        label_map = build_label_map(label_files, sample_meta, sample_name=sample['subfolder'], logger=LOGGER)
        total_frames = len(img_files)
        if total_frames == self.seq_length:
            selected_indices = list(range(self.seq_length))
        elif self.mode == 'train':
            label_indices = [idx for idx in label_map if 0 <= idx < total_frames]
            valid_starts = [
                start
                for start in range(0, total_frames - self.seq_length + 1)
                if any(start <= label_idx < start + self.seq_length for label_idx in label_indices)
            ]
            if valid_starts:
                start = int(valid_starts[int(torch.randint(0, len(valid_starts), (1,)).item())])
            else:
                start = int(torch.randint(0, total_frames - self.seq_length + 1, (1,)).item())
            selected_indices = list(range(start, start + self.seq_length))
        else:
            selected_indices = np.linspace(0, total_frames - 1, self.seq_length).round().astype(int).tolist()
        selected_to_local = {src_idx: local_idx for local_idx, src_idx in enumerate(selected_indices)}
        local_label_map = {
            selected_to_local[src_idx]: label_name
            for src_idx, label_name in label_map.items()
            if src_idx in selected_to_local
        }

        for i in range(self.seq_length):
            img_path = os.path.join(img_folder, img_files[selected_indices[i]])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                if img.shape != (self.size, self.size):
                    img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                imgs_np[i] = img
            
            mask_path = None
            if i in local_label_map:
                mask_path = os.path.join(label_folder, local_label_map[i])

            if mask_path:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape != (self.size, self.size):
                        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                    masks_np[i] = (mask > 0).astype(np.uint8)

        frames_t = torch.from_numpy(imgs_np).float().unsqueeze(1) / 255.0
        masks_t = torch.from_numpy(masks_np).long().unsqueeze(1)
        if self.mode == 'train':
            frames_t = _apply_intensity_augmentation(frames_t, self.augmentation)

        info = {
            'name': sample['subfolder'],
            'frames': img_files,
            'num_objects': 0
        }

        cls_gt = torch.zeros_like(masks_t)
        first_frame_gt = torch.zeros((1, self.max_num_obj, self.size, self.size), dtype=torch.long)
        selector = torch.zeros(self.max_num_obj, dtype=torch.float32)
        label_valid = torch.zeros(self.seq_length, dtype=torch.bool)
        eval_valid = torch.zeros(self.seq_length, dtype=torch.bool)

        frame_indices_all = sample_meta.get('source_frames', list(range(total_frames)))
        if len(frame_indices_all) >= total_frames:
            frame_indices = [frame_indices_all[idx] for idx in selected_indices]
        else:
            frame_indices = selected_indices
        original_size = sample_meta.get('original_size', [self.size, self.size])
        protocol_name = sample_meta.get('protocol_name', _infer_protocol_name(self.filepath))
        original_sizes = torch.tensor([original_size] * self.seq_length, dtype=torch.long)
        resized_sizes = torch.tensor([[self.size, self.size]] * self.seq_length, dtype=torch.long)

        has_foreground_label = masks_t.max() > 0
        if has_foreground_label:
            info['num_objects'] = 1
            selector[0] = 1.0
            
            cls_gt = masks_t.clone()
            if masks_t[0].max() > 0:
                first_frame_gt[0, 0] = masks_t[0, 0]
            for idx in local_label_map:
                if 0 <= idx < self.seq_length:
                    label_valid[idx] = True
                    eval_valid[idx] = True
        if not label_valid.any():
            raise ValueError(f"EchoDataset sample {sample['subfolder']} has no labels after frame selection")
        info['valid_label_frames'] = [int(idx) for idx in sorted(local_label_map) if 0 <= idx < self.seq_length]
        info['has_first_frame_gt'] = bool(masks_t[0].max() > 0)

        data = {
            'rgb': frames_t,
            'ff_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'label_valid': label_valid,
            'eval_valid': eval_valid,
            'selector': selector,
            'info': info,
            'original_size': original_sizes,
            'resized_size': resized_sizes,
            'frame_indices': torch.tensor(frame_indices, dtype=torch.long),
            'protocol_name': protocol_name,
        }

        return data
