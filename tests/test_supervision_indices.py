import unittest

import torch
from omegaconf import OmegaConf

from model.losses import LossComputer
from model.trainer import Trainer
from tests.factories import make_cls_gt_from_frame_labels, make_frame_valid_mask, make_video_batch


class SupervisionIndexTests(unittest.TestCase):
    def test_dense_label_valid_uses_all_frames(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        data = {
            "rgb": make_video_batch(2, 4),
            "label_valid": torch.ones(2, 4, dtype=torch.bool),
        }
        idx = Trainer._resolve_supervised_indices(trainer, data)
        self.assertTrue(torch.equal(idx, torch.ones(2, 4, dtype=torch.bool)))

    def test_sparse_label_valid_keeps_selected_frames_per_sample(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        data = {
            "rgb": make_video_batch(2, 5),
            "label_valid": make_frame_valid_mask(
                [True, False, False, False, True],
                [False, True, False, True, False],
            ),
        }
        idx = Trainer._resolve_supervised_indices(trainer, data)
        self.assertTrue(torch.equal(idx, data["label_valid"]))

    def test_eval_valid_keeps_selected_frames_per_sample(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = OmegaConf.create({"evaluation": {"frame_scope": "all_available"}})
        data = {
            "rgb": make_video_batch(2, 5),
            "label_valid": make_frame_valid_mask(
                [True, False, False, False, True],
                [False, True, False, True, False],
            ),
            "eval_valid": make_frame_valid_mask(
                [True, True, False, False, True],
                [False, True, True, True, False],
            ),
        }
        idx = Trainer._resolve_eval_indices(trainer, data)
        self.assertTrue(torch.equal(idx, data["eval_valid"]))

    def test_loss_computer_accepts_per_sample_supervision_masks(self):
        cfg = OmegaConf.create(
            {
                "model": {
                    "aux_loss": {
                        "sensory": {"weight": 0.0},
                        "query": {"weight": 0.0},
                    },
                    "temporal_memory": {"bpm": {}},
                }
            }
        )
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": True,
                "train_num_points": 4,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        loss_computer = LossComputer(cfg, stage_cfg)
        loss_computer.mask_loss = lambda logits, soft_gt: (
            torch.tensor(1.0, device=logits.device),
            torch.tensor(2.0, device=logits.device),
        )

        data = {
            "rgb": make_video_batch(2, 5),
            "cls_gt": make_cls_gt_from_frame_labels(
                [[0, 1, 0, 0, 1], [0, 0, 1, 1, 0]],
            ),
            "supervised_indices": make_frame_valid_mask(
                [True, True, False, False, True],
                [False, False, True, True, False],
            ),
            "logits_0": torch.randn(2, 2, 8, 8),
            "logits_1": torch.randn(2, 2, 8, 8),
            "logits_2": torch.randn(2, 2, 8, 8),
            "logits_3": torch.randn(2, 2, 8, 8),
            "logits_4": torch.randn(2, 2, 8, 8),
            "aux_0": {},
            "aux_1": {},
            "aux_2": {},
            "aux_3": {},
            "aux_4": {},
        }

        losses = loss_computer.compute(data, num_objects=[1, 1])
        self.assertIn("total_loss", losses)
        self.assertTrue(torch.isfinite(losses["total_loss"]))


if __name__ == "__main__":
    unittest.main()
