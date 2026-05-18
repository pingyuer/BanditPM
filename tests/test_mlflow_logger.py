import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from utils.mlflow_logger import MLflowLogger


class _FakeRun:
    info = types.SimpleNamespace(run_id="run-123")


class MLflowLoggerTests(unittest.TestCase):
    def test_start_resume_and_log_calls(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            set_tracking_uri=lambda uri: calls.append(("tracking_uri", uri)),
            set_experiment=lambda name: calls.append(("experiment", name)),
            start_run=lambda **kwargs: calls.append(("start_run", kwargs)) or _FakeRun(),
            end_run=lambda status="FINISHED": calls.append(("end_run", status)),
            log_params=lambda params: calls.append(("params", params)),
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
            log_artifact=lambda path, artifact_path=None: calls.append(("artifact", Path(path).name, artifact_path)),
            log_artifacts=lambda path, artifact_path=None: calls.append(("artifacts", Path(path).name, artifact_path)),
        )
        cfg = OmegaConf.create(
            {
                "enabled": True,
                "tracking_uri": "http://172.16.240.77:5000",
                "experiment_name": "anchor_ode",
                "run_name": None,
                "resume_run_id": "abc",
            }
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger(cfg, run_dir=tmp, enabled=True, main_process=True)
            logger.start_run()
            logger.log_config(OmegaConf.create({"model": {"name": "debug"}, "seed": 7}))
            logger.log_metrics({"dice": 0.9, "bad": float("nan")}, step=3, prefix="val")
            logger.mark_failed()

        self.assertIn(("tracking_uri", "http://172.16.240.77:5000"), calls)
        self.assertIn(("experiment", "anchor_ode"), calls)
        self.assertIn(("start_run", {"run_id": "abc"}), calls)
        self.assertIn(("metrics", {"val/dice": 0.9}, 3), calls)
        self.assertIn(("end_run", "FAILED"), calls)
        self.assertTrue(any(call[0] == "artifact" and call[2] == "configs" for call in calls))

    def test_non_main_process_is_noop(self):
        cfg = OmegaConf.create({"enabled": True})
        logger = MLflowLogger(cfg, run_dir=".", enabled=True, main_process=False)
        logger.start_run()
        logger.log_metrics({"loss": 1.0})
        self.assertIsNone(logger.run_id)


if __name__ == "__main__":
    unittest.main()
