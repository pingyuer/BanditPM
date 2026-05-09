import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.losses import LossComputer
from dataset.frame_index import build_label_map, parse_frame_index
from model.memory_readout import MaskAwareMemoryReadout
from model.modules.memory_core import MemoryCore
from model.modules.unext.unext import UNeXtBackbone
from model.unext_dynakey import UNeXtDynaKeySegmenter


def _cfg(*, temporal_refine=True, ode_aux=False, use_dynakey=True, use_mask_memory=True):
    return OmegaConf.create(
        {
            "model": {
                "aux_loss": {
                    "sensory": {"weight": 0.0},
                    "query": {"weight": 0.0},
                },
                "memory_core": {
                    "type": "dynakey",
                    "dynakey": {
                        "BANK_SIZE": 3,
                        "DT": 1.0,
                        "EMA_ALPHA": 0.3,
                        "HIDDEN_DIM": 32,
                        "GATE_INIT": 1.0,
                        "POLICY_MODE": "fixed_residual",
                        "ENABLE_Q_LOSS": False,
                    },
                },
                "temporal_memory": {"type": "dynakey", "bpm": {}},
                "unext_dynakey": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "base_dim": 8,
                    "value_dim": 16,
                    "use_dynakey": use_dynakey,
                    "use_temporal_refine": temporal_refine,
                    "use_mask_memory": use_mask_memory,
                    "temporal_residual_init_scale": 0.1,
                    "temporal_gate_bias": -2.0,
                    "mask_memory_confidence_threshold": 0.55,
                    "mask_memory_fg_ratio_min": 0.005,
                    "mask_memory_fg_ratio_max": 0.60,
                    "use_ode_aux_loss": ode_aux,
                },
            }
        }
    )


def _batch(batch_size=2, frames=3, height=32, width=32):
    rgb = torch.randn(batch_size, frames, 1, height, width)
    cls_gt = torch.randint(0, 2, (batch_size, frames, 1, height, width), dtype=torch.long)
    ff_gt = F.one_hot(cls_gt[:, :1, 0], num_classes=2)[..., 1:].permute(0, 1, 4, 2, 3).float()
    return {
        "rgb": rgb,
        "ff_gt": ff_gt,
            "cls_gt": cls_gt,
            "label_valid": torch.ones(batch_size, frames, dtype=torch.bool),
            "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
            "init_mode": "oracle_gt",
        }


class UNeXtDynaKeyTests(unittest.TestCase):
    def test_frame_index_parser_common_echo_names(self):
        metadata = {"ED_frame": 0, "ES_frame": 9}
        self.assertEqual(parse_frame_index("000.png"), 0)
        self.assertEqual(parse_frame_index("000001_mask.png"), 1)
        self.assertEqual(parse_frame_index("frame_007.png"), 7)
        self.assertEqual(parse_frame_index("frame001.png"), 1)
        self.assertEqual(parse_frame_index("ED.png", metadata), 0)
        self.assertEqual(parse_frame_index("ES.png", metadata), 9)
        self.assertIsNone(parse_frame_index("not_a_frame.png"))

    def test_frame_index_maps_source_frames_to_local_indices(self):
        metadata = {"source_frames": [30, 35, 40, 45, 50], "ED_frame": 40, "ES_frame": 50}
        label_map = build_label_map(["ED.png", "ES.png", "frame_040.png"], metadata, sample_name="toy")
        self.assertEqual(label_map[2], "ED.png")
        self.assertEqual(label_map[4], "ES.png")

    def test_unext_backbone_shapes(self):
        model = UNeXtBackbone(in_channels=1, num_classes=2, base_dim=8, value_dim=16)
        out = model(torch.randn(2, 1, 32, 32))
        self.assertEqual(out["logits"].shape, (2, 2, 32, 32))
        self.assertEqual(out["low"].shape[-2:], (16, 16))
        self.assertEqual(out["mid"].shape[-2:], (8, 8))
        self.assertEqual(out["high"].shape[-2:], (4, 4))
        self.assertEqual(out["value"].shape, (2, 16, 8, 8))
        self.assertEqual(out["decoder_feature"].shape[-2:], (32, 32))

    def test_unext_dynakey_forward_with_temporal_refine(self):
        model = UNeXtDynaKeySegmenter(_cfg(temporal_refine=True).model)
        data = _batch()
        out = model(data)
        for t in range(3):
            self.assertEqual(out[f"logits_{t}"].shape, (2, 2, 32, 32))
            self.assertEqual(out[f"masks_{t}"].shape, (2, 1, 32, 32))
            self.assertIn("dynakey_aux", out[f"memory_aux_{t}"])
        self.assertEqual(out["num_objects"], [1, 1])

    def test_unext_dynakey_forward_without_temporal_refine(self):
        model = UNeXtDynaKeySegmenter(_cfg(temporal_refine=False).model)
        out = model(_batch(batch_size=1))
        self.assertEqual(out["logits_1"].shape, (1, 2, 32, 32))
        self.assertTrue(torch.isfinite(out["logits_1"]).all())

    def test_unext_only_and_temporal_refine_only_modes(self):
        unext_only = UNeXtDynaKeySegmenter(_cfg(temporal_refine=False, use_dynakey=False, use_mask_memory=False).model)
        out = unext_only(_batch(batch_size=1))
        self.assertEqual(out["memory_aux_1"]["memory_type"], "none")
        self.assertFalse(out["memory_aux_1"]["dynakey_enabled"])
        self.assertFalse(out["memory_aux_1"]["temporal_refine_enabled"])
        self.assertTrue(torch.allclose(out["aux_1"]["object_logits"], out["aux_1"]["base_foreground_logits"], atol=1e-6))

        temporal_only = UNeXtDynaKeySegmenter(_cfg(temporal_refine=True, use_dynakey=False, use_mask_memory=False).model)
        out = temporal_only(_batch(batch_size=1))
        self.assertEqual(out["logits_2"].shape, (1, 2, 32, 32))
        self.assertFalse(out["memory_aux_2"]["dynakey_enabled"])

    def test_unext_dynakey_backward_has_expected_grads(self):
        cfg = _cfg(temporal_refine=True)
        model = UNeXtDynaKeySegmenter(cfg.model)
        data = _batch(batch_size=2, frames=3)
        out = model(data)
        data.update(out)
        data["supervised_indices"] = torch.ones(2, 3, dtype=torch.bool)
        loss_cfg = cfg
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": True,
                "train_num_points": 64,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        loss = LossComputer(loss_cfg, stage_cfg).compute(data, [1, 1])["total_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.backbone.input_down[0].weight.grad)
        self.assertIsNotNone(model.temporal_refine_head[0].weight.grad)
        self.assertIsNotNone(model.temporal_delta_proj.weight.grad)
        self.assertIsNotNone(model.temporal_gate_head[0].weight.grad)

    def test_no_leak_pred_or_zero_ignores_first_frame_gt(self):
        cfg = _cfg(temporal_refine=False, use_dynakey=False, use_mask_memory=False)
        cfg.model.allow_oracle_init_when_requested = False
        model = UNeXtDynaKeySegmenter(cfg.model)
        data = _batch(batch_size=1)
        data["init_mode"] = "pred_or_zero"
        out = model(data)
        self.assertFalse(out["memory_aux_0"]["oracle_gt_init_allowed"])
        self.assertTrue(torch.allclose(out["aux_0"]["object_logits"], out["aux_0"]["base_foreground_logits"], atol=1e-6))

    def test_mask_memory_rejects_empty_and_huge_predictions(self):
        mem = MaskAwareMemoryReadout(4, confidence_threshold=0.55, fg_ratio_min=0.05, fg_ratio_max=0.60)
        value = torch.randn(1, 1, 4, 4, 4)
        empty = torch.zeros(1, 1, 16, 16)
        aux_empty = mem.update(value, empty)
        self.assertEqual(int(aux_empty["rejected_update_too_small"].item()), 1)
        self.assertEqual(int(mem._valid.sum().item()), 0)
        huge = torch.ones(1, 1, 16, 16)
        aux_huge = mem.update(value, huge)
        self.assertEqual(int(aux_huge["rejected_update_too_large"].item()), 1)
        self.assertEqual(int(mem._valid.sum().item()), 0)

    def test_existing_dynakey_memory_core_still_runs(self):
        cfg = OmegaConf.create(
            {
                "type": "dynakey",
                "dynakey": {"BANK_SIZE": 2, "HIDDEN_DIM": 16},
            }
        )
        core = MemoryCore(
            value_dim=4,
            key_dim=2,
            temporal_memory_cfg=OmegaConf.create({"type": "dynakey", "dynakey": cfg.dynakey}),
            memory_core_cfg=cfg,
        )
        core.reset_state(1, 1, torch.device("cpu"))
        readout, aux = core(
            torch.randn(1, 1, 4, 2, 2),
            torch.randn(1, 2, 2, 2),
            torch.randn(1, 4, 2, 2),
            torch.ones(1, 1, 8, 8),
        )
        self.assertEqual(readout.shape, (1, 1, 4, 2, 2))
        self.assertEqual(aux["memory_type"], "dynakey")


if __name__ == "__main__":
    unittest.main()
