from pathlib import Path

from omegaconf import OmegaConf

from experiment import MLflowLogger


cfg = OmegaConf.create(
    {
        "enabled": True,
        "tracking_uri": "http://172.16.240.77:5000",
        "experiment_name": "test-mlflow-garage",
        "run_name": "debug",
        "resume_run_id": None,
    }
)

Path("hello.txt").write_text("hello mlflow garage\n", encoding="utf-8")

logger = MLflowLogger(cfg, run_dir=".", enabled=True, main_process=True)
logger.start_run()
try:
    logger.log_params({"model": "debug"})
    logger.log_metrics({"miou": 0.731})
    logger.log_artifact("hello.txt")
    logger.end_run()
except Exception:
    logger.mark_failed()
    raise
