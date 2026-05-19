import unittest

import torch
from omegaconf import OmegaConf

from losses import LossComputer
from training import Trainer
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

    def test_eval_exclude_init_frame_removes_frame_zero(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = OmegaConf.create(
            {
                "evaluation": {
                    "frame_scope": "all_available",
                    "exclude_init_frame": True,
                    "init_frame_index": 0,
                }
            }
        )
        data = {
            "rgb": make_video_batch(2, 5),
            "label_valid": make_frame_valid_mask(
                [True, False, False, False, True],
                [True, True, False, True, False],
            ),
            "eval_valid": make_frame_valid_mask(
                [True, True, False, False, True],
                [True, True, True, True, False],
            ),
        }
        idx = Trainer._resolve_eval_indices(trainer, data)
        expected = make_frame_valid_mask(
            [False, True, False, False, True],
            [False, True, True, True, False],
        )
        self.assertTrue(torch.equal(idx, expected))

    def test_eval_exclude_init_frame_does_not_fallback_to_leaky_init(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = OmegaConf.create(
            {
                "evaluation": {
                    "frame_scope": "supervised_only",
                    "exclude_init_frame": True,
                    "init_frame_index": 0,
                }
            }
        )
        data = {
            "rgb": make_video_batch(2, 4),
            "label_valid": make_frame_valid_mask(
                [True, False, False, False],
                [True, False, False, False],
            ),
        }
        idx = Trainer._resolve_eval_indices(trainer, data)
        self.assertFalse(idx.any())

    def test_summary_row_omits_protocol_version_from_experiment_summary(self):
        trainer = Trainer.__new__(Trainer)
        trainer.exp_id = "unit_no_leak"
        trainer.commit_hash = "test"
        trainer.cfg = OmegaConf.create(
            {
                "dataset_name": "echonet",
                "data": {"protocol_name": "unit"},
                "seed": 7,
                "phase_init": {"test": "pred_or_zero"},
                "evaluation": {
                    "frame_scope": "supervised_only",
                    "init_mode": "pred_or_zero",
                    "exclude_init_frame": True,
                    "init_frame_index": 0,
                    "protocol_version": "v2_no_leak",
                },
            }
        )
        row = Trainer._build_summary_row(trainer, "test", {}, epoch=0, it=0)
        self.assertEqual(row["init_mode"], "pred_or_zero")
        self.assertTrue(row["exclude_init_frame"])
        self.assertNotIn("protocol_version", row)

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
