import importlib


def test_public_package_boundaries_are_importable():
    from experiment import MLflowLogger
    from losses import LossComputer
    from models.registry import MODEL_REGISTRY, build_model
    from training import ModelEMA, Trainer, TrainingLogger
    from visualization import visualize_sequence

    assert MLflowLogger is not None
    assert LossComputer is not None
    assert MODEL_REGISTRY is not None
    assert build_model is not None
    assert ModelEMA is not None
    assert Trainer is not None
    assert TrainingLogger is not None
    assert visualize_sequence is not None


def test_removed_legacy_shim_modules_are_not_importable():
    removed_modules = [
        "model.trainer",
        "model.losses",
        "model.registry",
        "utils.mlflow_logger",
        "utils.logger",
        "vis.vis_0730",
    ]
    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None
