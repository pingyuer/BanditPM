from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_remaining_canonical_configs_compose_and_are_no_leak():
    names = [p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem.startswith(("gdkvm_", "dpfr_"))]
    assert names
    with initialize_config_dir(version_base="1.3.2", config_dir=str(CONFIG_DIR)):
        for name in names:
            cfg = compose(config_name=name)
            assert str(cfg.model.name).lower() in {"gdkvm", "dpfr"}
            assert str(cfg.evaluation.init_mode) == "pred_or_zero"
            assert bool(cfg.evaluation.exclude_init_frame)
            assert str(cfg.evaluation.protocol_version) == "v3_canonical_no_leak"
