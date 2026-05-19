import unittest
from unittest import mock

from omegaconf import OmegaConf

from experiment.metadata import build_mlflow_metadata


class MLflowMetadataTests(unittest.TestCase):
    def test_builds_search_tags_and_core_params(self):
        cfg = OmegaConf.create(
            {
                "model_name": "gdkvm",
                "dataset_name": "echonet",
                "exp_id": "gdkvm_echo",
                "seed": 11,
                "data": {"protocol_name": "ed2es"},
                "mlflow": {"experiment_name": None},
                "model": {"name": "gdkvm", "version": "do_not_log", "memory_core": {"type": "original_gdr"}},
                "model_version": "do_not_log",
                "main_training": {
                    "num_iterations": 100,
                    "learning_rate": 1.0e-4,
                    "batch_size": 8,
                    "seq_length": 10,
                },
                "evaluation": {"protocol_version": "v3"},
            }
        )
        with mock.patch(
            "experiment.metadata.resolve_git_metadata",
            return_value={"git_commit": "abc", "git_short": "abc", "git_dirty": True},
        ):
            tags, params = build_mlflow_metadata(cfg, world_size=2)
        self.assertEqual(tags["run_type"], "train")
        self.assertEqual(tags["project"], "tahara-3d")
        self.assertEqual(tags["method"], "gdkvm")
        self.assertEqual(tags["protocol"], "ed2es")
        self.assertEqual(tags["stage"], "full")
        self.assertEqual(tags["ddp_world_size"], 2)
        self.assertEqual(tags["git_commit"], "abc")
        self.assertEqual(tags["git_dirty"], True)
        self.assertEqual(params["train.lr"], 1.0e-4)
        self.assertEqual(params["model.name"], "gdkvm")
        self.assertNotIn("model.version", params)


if __name__ == "__main__":
    unittest.main()
