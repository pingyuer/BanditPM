import unittest

import torch
from omegaconf import OmegaConf

from training import Trainer


def _trainer(metric_space="original"):
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.is_distributed = False
    trainer.cfg = OmegaConf.create({"evaluation": {"metric_space": metric_space}})
    return trainer


class PhaseMetricTests(unittest.TestCase):
    def test_dense_sequence_uses_first_and_last_eval_frame_as_ed_es(self):
        trainer = _trainer()
        totals = trainer._metric_totals_template()
        totals.update(
            {
                "dice_frame_sum": 2.4,
                "dice_frame_count": 3.0,
                "hd95_original_sum": 12.0,
                "hd95_original_count": 3.0,
                "ed_dice_sum": 0.9,
                "ed_dice_count": 1.0,
                "es_dice_sum": 0.7,
                "es_dice_count": 1.0,
                "ed_hd95_original_sum": 2.0,
                "ed_hd95_count": 1.0,
                "es_hd95_original_sum": 6.0,
                "es_hd95_count": 1.0,
            }
        )
        metrics = trainer._reduce_metric_totals(totals)
        self.assertAlmostEqual(metrics["ed_dice"], 0.9)
        self.assertAlmostEqual(metrics["es_dice"], 0.7)
        self.assertAlmostEqual(metrics["ed_hd95"], 2.0)
        self.assertAlmostEqual(metrics["es_hd95"], 6.0)
        self.assertAlmostEqual(metrics["overall_dice"], metrics["dice"])
        self.assertAlmostEqual(metrics["overall_hd95"], metrics["hd95"])

    def test_resolve_phase_frames_uses_explicit_ed_es_when_available(self):
        trainer = _trainer()
        batch = {"ed_frame": torch.tensor([1]), "es_frame": torch.tensor([3])}
        self.assertEqual(trainer._resolve_phase_eval_frames(batch, 0, [0, 1, 2, 3]), (1, 3))

    def test_resolve_phase_frames_falls_back_to_valid_endpoints(self):
        trainer = _trainer()
        self.assertEqual(trainer._resolve_phase_eval_frames({}, 0, [2, 4, 5]), (2, 5))

    def test_excluded_explicit_ed_is_not_reassigned(self):
        trainer = _trainer()
        batch = {"ed_frame": torch.tensor([0]), "es_frame": torch.tensor([3])}
        self.assertEqual(trainer._resolve_phase_eval_frames(batch, 0, [1, 2, 3]), (None, 3))

    def test_metric_space_controls_phase_hd95_alias(self):
        trainer = _trainer(metric_space="resized")
        totals = trainer._metric_totals_template()
        totals.update(
            {
                "dice_frame_sum": 1.0,
                "dice_frame_count": 1.0,
                "hd95_resized_sum": 4.0,
                "hd95_resized_count": 1.0,
                "hd95_original_sum": 40.0,
                "hd95_original_count": 1.0,
                "ed_hd95_resized_sum": 3.0,
                "ed_hd95_original_sum": 30.0,
                "ed_hd95_count": 1.0,
            }
        )
        metrics = trainer._reduce_metric_totals(totals)
        self.assertAlmostEqual(metrics["ed_hd95"], 3.0)
        self.assertAlmostEqual(metrics["overall_hd95"], 4.0)


if __name__ == "__main__":
    unittest.main()
