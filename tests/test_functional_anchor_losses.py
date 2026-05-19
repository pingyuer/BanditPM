import unittest

import torch
from omegaconf import OmegaConf

from model.functional_anchor import FunctionalAnchorSegmenter
from model.functional_anchor.anchor_bank import FunctionalAnchorBank
from losses import LossComputer


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "functional_anchor",
                "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
                "memory_core": {"type": "none", "dynakey": {}},
                "temporal_memory": {"type": "none", "bpm": {}},
                "functional_anchor": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "base_dim": 8,
                    "value_dim": 16,
                    "num_slots": 5,
                    "state_dim": 24,
                    "phase_dim": 8,
                    "hidden_dim": 32,
                    "anchor_size": 8,
                    "residual_clip": 1.0,
                    "prediction_mode": "base_primary",
                    "lambda_anchor": 0.5,
                    "lambda_base_seg": 0.1,
                    "lambda_residual_smallness": 0.02,
                    "lambda_boundary_residual": 0.1,
                    "lambda_phase_consistency": 0.02,
                    "lambda_anchor_temporal": 0.02,
                    "lambda_slot_area_order": 0.01,
                    "lambda_phase_slot_correlation": 0.01,
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


class FunctionalAnchorLossTests(unittest.TestCase):
    def test_loss_computer_emits_functional_anchor_terms(self):
        torch.manual_seed(201)
        cfg = _cfg()
        model = FunctionalAnchorSegmenter(cfg.model)
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
            "aux_functional_anchor_anchor",
            "aux_functional_anchor_base",
            "aux_functional_anchor_residual_l1",
            "aux_functional_anchor_boundary_residual",
            "aux_functional_anchor_phase_consistency",
            "aux_functional_anchor_anchor_temporal",
            "aux_functional_anchor_slot_area_order",
            "aux_functional_anchor_phase_slot_correlation",
        ):
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]))
        self.assertTrue(torch.isfinite(losses["total_loss"]))

    def test_slot_semantic_loss_penalizes_area_order_violation(self):
        cfg = _cfg()
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 64,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        loss_computer = LossComputer(cfg, stage_cfg)
        violation = torch.tensor([[0.3], [0.1]])
        data = _batch(batch_size=2, frames=1)
        data["memory_aux_0"] = {
            "functional_anchor_aux": {
                "anchor_logits": torch.zeros(2, 1, 32, 32),
                "base_object_logits": torch.zeros(2, 1, 32, 32),
                "final_object_logits": torch.zeros(2, 1, 32, 32),
                "residual_logits": torch.zeros(2, 1, 32, 32),
                "slot_area_order_violation": violation,
                "slot_weights": torch.full((2, 1, 5), 0.2),
                "phase_descriptor": torch.zeros(2, 1, 17),
            }
        }
        data["logits_0"] = torch.zeros(2, 2, 32, 32)
        terms = loss_computer._compute_functional_anchor_losses(data, torch.ones(2, 1, dtype=torch.bool))
        self.assertIn("aux_functional_anchor_slot_area_order", terms)
        self.assertGreater(float(terms["aux_functional_anchor_slot_area_order"]), 0.0)

    def test_slot_order_is_cardiac_cycle_structured_not_global_monotonic(self):
        bank = FunctionalAnchorBank(num_slots=5, state_dim=8, phase_dim=4, hidden_dim=12)
        with torch.no_grad():
            areas = torch.tensor([0.85, 0.62, 0.30, 0.58, 0.95])
            bank.area_bias.copy_(torch.logit(areas))
        z = torch.zeros(1, 1, 8)
        phase = torch.zeros(1, 1, 4)
        _, aux = bank(z, phase, torch.zeros(1, 1))
        self.assertLess(float(aux["slot_order_loss"]), 1.0e-6)
        self.assertGreater(float(aux["slot_area_uncertain"]), float(aux["slot_area_ed"]))


if __name__ == "__main__":
    unittest.main()
