import unittest

import torch
from omegaconf import OmegaConf

from losses import LossComputer
from model.functional_anchor.metrics import dice_per_anchor
from model.unext_faf import UNeXtFAF


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "unext_faf",
                "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
                "memory_core": {"type": "none", "dynakey": {}},
                "temporal_memory": {"type": "none", "bpm": {}},
                "unext_faf": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "base_dim": 8,
                    "value_dim": 16,
                    "num_anchors": 4,
                    "num_affine_slots": 4,
                    "identity_slot_index": 0,
                    "query_dim": 16,
                    "code_dim": 16,
                    "hidden_dim": 24,
                    "basis_dim": 4,
                    "anchor_size": 8,
                    "residual_clip": 0.5,
                    "trust_max": 0.6,
                    "retrieval_temperature": 0.4,
                    "memory_ema": 0.9,
                    "prediction_mode": "affine_mixture_safe",
                    "require_pretrained_unext": False,
                    "selector": {"temperature_init": 1.0, "temperature_final": 0.35, "assignment_temperature": 0.15},
                    "confidence": {"enabled": True, "init": 0.10, "max": 0.35, "warmup_iters": 1500},
                    "residual": {"enabled": True, "init_scale": 0.0, "max_scale": 0.05, "clip": 0.15},
                    "temporal_update": {"enabled": True, "dt_init": 0.1, "dt_max": 0.6, "truncated_bptt_steps": 0},
                    "lambda_faf_affine": 0.001,
                    "lambda_faf_velocity": 0.001,
                    "lambda_faf_mixture": 0.3,
                    "lambda_faf_oracle": 0.2,
                    "lambda_faf_top1": 0.1,
                    "lambda_faf_selector": 0.1,
                    "lambda_faf_confidence": 0.05,
                    "lambda_faf_base": 1.0,
                    "lambda_faf_coverage": 0.05,
                    "lambda_faf_sparse": 0.002,
                    "lambda_faf_diversity": 0.001,
                    "lambda_faf_temporal": 0.02,
                    "lambda_faf_write": 0.001,
                    "lambda_faf_residual_smallness": 0.05,
                    "lambda_faf_feature_modulation": 0.001,
                },
            }
        }
    )


def _batch(batch_size=2, frames=3, height=32, width=32):
    return {
        "rgb": torch.rand(batch_size, frames, 1, height, width),
        "cls_gt": torch.randint(0, 2, (batch_size, frames, 1, height, width)),
        "supervised_indices": torch.ones(batch_size, frames, dtype=torch.bool),
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
    }


class FAFLossTests(unittest.TestCase):
    def test_loss_computer_emits_faf_terms(self):
        torch.manual_seed(401)
        cfg = _cfg()
        model = UNeXtFAF(cfg.model)
        data = _batch(batch_size=2, frames=3)
        data.update(model(data))
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 64,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        for key in (
            "aux_faf_oracle",
            "aux_faf_top1",
            "aux_faf_mixture",
            "aux_faf_selector",
            "aux_faf_confidence",
            "aux_faf_base",
            "aux_faf_coverage",
            "aux_faf_sparse",
            "aux_faf_diversity",
            "aux_faf_temporal",
            "aux_faf_write",
            "aux_faf_residual_smallness",
            "aux_faf_affine",
            "aux_faf_velocity",
        ):
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]))
        self.assertTrue(torch.isfinite(losses["total_loss"]))

    def test_affine_regularizer_has_finite_grad_at_zero_init(self):
        torch.manual_seed(402)
        cfg = _cfg()
        model = UNeXtFAF(cfg.model)
        data = _batch(batch_size=1, frames=2)
        data.update(model(data))
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 64,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        losses = LossComputer(cfg, stage_cfg).compute(data, [1])

        self.assertIn("aux_faf_affine", losses)
        self.assertTrue(torch.isfinite(losses["aux_faf_affine"]))
        losses["aux_faf_affine"].backward()
        bad_grads = [
            name
            for name, param in model.named_parameters()
            if param.grad is not None and not torch.isfinite(param.grad).all()
        ]
        self.assertEqual(bad_grads, [])

    def test_proposal_oracle_dice_separates_assignment_from_coverage(self):
        gt = torch.zeros(1, 1, 8, 8)
        gt[:, :, 2:6, 2:6] = 1.0
        proposals = torch.full((1, 1, 3, 8, 8), -6.0)
        proposals[:, :, 0, :2, :2] = 6.0
        proposals[:, :, 1, 2:6, 2:6] = 6.0
        proposals[:, :, 2, 5:, 5:] = 6.0
        dice = dice_per_anchor(proposals, gt)
        top1_dice = dice[..., 0]
        oracle_dice = dice.max(dim=-1).values
        self.assertLess(float(top1_dice.mean()), 0.5)
        self.assertGreater(float(oracle_dice.mean()), 0.9)


if __name__ == "__main__":
    unittest.main()
