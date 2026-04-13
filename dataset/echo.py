import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def _infer_protocol_name(filepath: str) -> str:
    lower = filepath.lower()
    if "cardiacuda" in lower:
        return "cardiacuda_a4c_lv_sparse"
    if "full_cycle" in lower:
        return "echonet_fullcycle_sparse"
    return "echonet_ed2es_endpoint"

class EchoDataset(Dataset):
    """EchoNet-style sparse video dataset with keyframe supervision."""

    def __init__(self, filepath: str, mode: str = 'train', seq_length=10, max_num_obj=1, size=128, merge_probability=0.0):
        super().__init__()
        self.filepath = filepath
        self.mode = mode
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size
        self.merge_probability = merge_probability

        self.img_root = os.path.join(filepath, mode, 'img')
        self.label_root = os.path.join(filepath, mode, 'label')
        
        self.samples = []
        
        if os.path.isdir(self.img_root) and os.path.isdir(self.label_root):
            subfolders = sorted(os.listdir(self.img_root))
            
            for subfolder in subfolders:
                img_folder = os.path.join(self.img_root, subfolder)
                label_folder = os.path.join(self.label_root, subfolder)
                
                if os.path.isdir(img_folder) and os.path.isdir(label_folder):
                    img_files = sorted(os.listdir(img_folder))
                    label_files = sorted(os.listdir(label_folder))

                    if len(img_files) == self.seq_length and 1 <= len(label_files) <= self.seq_length:
                        self.samples.append({
                            'subfolder': subfolder,
                            'img_folder': img_folder,
                            'label_folder': label_folder,
                            'img_files': img_files,
                            'label_files': label_files,
                            'meta_path': os.path.join(filepath, mode, 'metadata', f'{subfolder}.json'),
                        })
        
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

        label_map = {}
        for label_name in label_files:
            stem = os.path.splitext(label_name)[0]
            if stem.isdigit():
                label_map[int(stem)] = label_name

        for i in range(self.seq_length):
            img_path = os.path.join(img_folder, img_files[i])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                if img.shape != (self.size, self.size):
                    img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                imgs_np[i] = img
            
            mask_path = None
            if i in label_map:
                mask_path = os.path.join(label_folder, label_map[i])

            if mask_path:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape != (self.size, self.size):
                        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                    masks_np[i] = (mask == 1).astype(np.uint8)

        frames_t = torch.from_numpy(imgs_np).float().unsqueeze(1) / 255.0
        masks_t = torch.from_numpy(masks_np).long().unsqueeze(1)

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

        frame_indices = sample_meta.get('source_frames', list(range(self.seq_length)))
        original_size = sample_meta.get('original_size', [self.size, self.size])
        protocol_name = sample_meta.get('protocol_name', _infer_protocol_name(self.filepath))
        original_sizes = torch.tensor([original_size] * self.seq_length, dtype=torch.long)
        resized_sizes = torch.tensor([[self.size, self.size]] * self.seq_length, dtype=torch.long)

        if masks_t[0].max() > 0:
            info['num_objects'] = 1
            selector[0] = 1.0
            
            cls_gt = masks_t.clone()
            first_frame_gt[0, 0] = masks_t[0, 0]
            for idx in label_map:
                if 0 <= idx < self.seq_length:
                    label_valid[idx] = True
                    eval_valid[idx] = True

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
