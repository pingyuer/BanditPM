import unittest

from omegaconf import OmegaConf

from utils.training_setup import scale_stage_for_world_size


class TrainingSetupTests(unittest.TestCase):
    def test_scale_stage_for_world_size_keeps_positive_batch_and_workers(self):
        stage_cfg = OmegaConf.create({"batch_size": 12, "num_workers": 8})
        scale_stage_for_world_size(stage_cfg, world_size=4)
        self.assertEqual(stage_cfg.batch_size, 3)
        self.assertEqual(stage_cfg.num_workers, 2)

    def test_scale_stage_for_world_size_clamps_to_one(self):
        stage_cfg = OmegaConf.create({"batch_size": 1, "num_workers": 1})
        scale_stage_for_world_size(stage_cfg, world_size=8)
        self.assertEqual(stage_cfg.batch_size, 1)
        self.assertEqual(stage_cfg.num_workers, 1)


if __name__ == "__main__":
    unittest.main()
