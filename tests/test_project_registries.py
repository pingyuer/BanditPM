import pytest
import torch
import importlib.util
from omegaconf import OmegaConf

from gdkvm_project.evaluation import METRIC_COLLECTOR_REGISTRY
from gdkvm_project.losses import LOSS_REGISTRY
from gdkvm_project.models import MODEL_REGISTRY, build_model
from gdkvm_project.visualization import VISUALIZER_REGISTRY
from model.gdkvm01 import GDKVM
from dpfr import DPFRSegmenter


def _gdkvm_cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "gdkvm",
                "allow_oracle_init_when_requested": False,
                "use_kpff": True,
                "backbone_pretrained": False,
                "memory_core": {"type": "none"},
                "temporal_memory": {"type": "none", "bpm": {}},
                "prototype_value": {"enable": False},
            }
        }
    )


def _dpfr_cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "dpfr",
                "dpfr": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "d_model": 24,
                    "temporal_layers": 1,
                    "temporal_heads": 4,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "max_time": 4,
                    "image_pool_hw": [1, 2],
                    "mask_pool_hw": [1, 2],
                    "prompt_injection": "gated_add",
                    "prompt_scales": ["low", "mid", "high"],
                    "output_init_std": 1.0e-5,
                    "flow_steps": 1,
                    "flow_context_channels": 8,
                    "flow_hidden_channels": 16,
                    "max_disp": 0.05,
                    "align_corners": True,
                    "padding_mode": "border",
                    "final_fusion": {
                        "gate_init": -2.0,
                        "max_prompt_scale": 0.5,
                        "max_flow_scale": 0.5,
                    },
                    "mask_prompt_train": {
                        "use_gt": True,
                        "gt_prob_start": 1.0,
                        "gt_prob_end": 1.0,
                        "schedule_iters": 1,
                        "mask_prob": 0.0,
                    },
                    "mask_prompt_eval": {"use_gt": False, "source": "anchor_or_mask"},
                    "backbone": {
                        "name": "official",
                        "base_dim": 8,
                        "mlp_expansion": 2.0,
                        "latent_blocks": 1,
                        "decoder_mlp_blocks": 1,
                    },
                },
            }
        }
    )


def test_public_model_registry_only_builds_gdkvm_and_dpfr():
    assert isinstance(build_model(_gdkvm_cfg(), device="cpu"), GDKVM)
    assert isinstance(build_model(_dpfr_cfg(), device="cpu"), DPFRSegmenter)
    for name in ["kpff", "unext_fusion", "delay_ode", "cardia", "rebel", "debel", "geomaskformer"]:
        cfg = OmegaConf.create({"model": {"name": name}})
        with pytest.raises(KeyError):
            MODEL_REGISTRY.build(cfg, device=torch.device("cpu"))


def test_extension_registries_accept_future_plugins():
    @LOSS_REGISTRY.register("toy_loss")
    def toy_loss(cfg):
        return cfg

    @METRIC_COLLECTOR_REGISTRY.register("toy_metrics")
    def toy_metrics(cfg, outputs):
        return outputs

    @VISUALIZER_REGISTRY.register("toy_visualizer")
    def toy_visualizer(cfg, batch):
        return batch

    assert LOSS_REGISTRY.build({"name": "toy_loss", "value": 1})["value"] == 1
    assert METRIC_COLLECTOR_REGISTRY.build({"name": "toy_metrics"}, outputs={"ok": True})["ok"]
    assert VISUALIZER_REGISTRY.build({"name": "toy_visualizer"}, batch={"ok": True})["ok"]


def test_removed_legacy_method_packages_are_not_importable():
    for module_name in ("cardia", "rebel", "debel", "geomaskformer", "model.functional_anchor", "model.modules.dynakey"):
        assert importlib.util.find_spec(module_name) is None
