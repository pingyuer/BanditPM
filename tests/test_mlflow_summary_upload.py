import tempfile
import unittest
import csv
from pathlib import Path

from training import Trainer
from scripts.summarize_and_clean_outputs import FIELDS, summarize


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

        self.assertEqual(logger.calls, [("summary.json", "eval"), ("summary.csv", "eval")])

    def test_missing_summary_is_noop(self):
        trainer = Trainer.__new__(Trainer)
        trainer.main_process = True
        logger = _Logger()
        trainer.mlflow_logger = logger
        with tempfile.TemporaryDirectory() as tmp:
            trainer.run_path = Path(tmp)
            trainer.upload_summary_artifact()

        self.assertEqual(logger.calls, [])

    def test_legacy_experiment_summary_has_no_version_column(self):
        self.assertNotIn("version", FIELDS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "summary.csv").write_text(
                "mode,experiment_name,dataset,protocol_name,protocol_version,iteration,dice_frame_mean\n"
                "test,exp,echo,ed2es,old_version,3,0.8\n",
                encoding="utf-8",
            )
            output = root / "EXPERIMENT_SUMMARY.csv"
            rows = summarize(root, output)

            self.assertEqual(len(rows), 1)
            self.assertNotIn("version", rows[0])
            with output.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            self.assertNotIn("version", header)


if __name__ == "__main__":
    unittest.main()
