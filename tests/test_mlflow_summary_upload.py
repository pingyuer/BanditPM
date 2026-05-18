import tempfile
import unittest
from pathlib import Path

from model.trainer import Trainer


class _Logger:
    def __init__(self):
        self.calls = []

    def log_artifact(self, path, artifact_path=None):
        self.calls.append((Path(path).name, artifact_path))


class MLflowSummaryUploadTests(unittest.TestCase):
    def test_summary_upload_is_explicit_end_of_training_action(self):
        trainer = Trainer.__new__(Trainer)
        trainer.main_process = True
        logger = _Logger()
        trainer.mlflow_logger = logger
        with tempfile.TemporaryDirectory() as tmp:
            trainer.run_path = Path(tmp)
            (trainer.run_path / "summary.csv").write_text("mode,dice\nval,0.9\n", encoding="utf-8")
            trainer.upload_summary_artifact()

        self.assertEqual(logger.calls, [("summary.csv", "eval")])

    def test_missing_summary_is_noop(self):
        trainer = Trainer.__new__(Trainer)
        trainer.main_process = True
        logger = _Logger()
        trainer.mlflow_logger = logger
        with tempfile.TemporaryDirectory() as tmp:
            trainer.run_path = Path(tmp)
            trainer.upload_summary_artifact()

        self.assertEqual(logger.calls, [])


if __name__ == "__main__":
    unittest.main()
