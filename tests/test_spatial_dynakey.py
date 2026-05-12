import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.losses import LossComputer
from model.spatial_dynakey import SpatialDynaKeyMemory, segmentation_gain_reward
from model.unext_dynakey import UNeXtDynaKeySegmenter


def _cfg(
    *,
    use_dynakey=True,
    memory_mode="spatial",
    refine=True,
    q_mode="off",
    q_reward_type="segmentation_gain",
    dynamics="spatial",
    phase=True,
    memory_fusion_level="late",
    prediction_mode="base_residual",
    base_logits_weight=1.0,
    detach_base_logits=False,
    base_prob_dropout=0.0,
    decoder_feature_dropout=0.0,
    freeze_unext=False,
    memory_only=False,
    lambda_memory_only=0.0,
):
    return OmegaConf.create(
        {
            "model": {
                "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
                "memory_core": {
                    "type": "dynakey",
                    "dynakey": {
                        "BANK_SIZE": 3,
                        "HIDDEN_DIM": 16,
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
                    "dynakey_memory_mode": memory_mode,
                    "use_phase_retrieval": phase,
                    "readout_type": "spatial_gate" if memory_mode == "spatial" else "global_broadcast",
                    "dynamics_mode": dynamics,
                    "memory_fusion_level": memory_fusion_level,
                    "prediction_mode": prediction_mode,
                    "base_logits_weight": base_logits_weight,
                    "detach_base_logits": detach_base_logits,
                    "base_prob_dropout": base_prob_dropout,
                    "decoder_feature_dropout": decoder_feature_dropout,
                    "freeze_unext": freeze_unext,
                    "memory_only_head_enabled": memory_only,
                    "lambda_memory_only": lambda_memory_only,
                    "q_policy_mode": q_mode,
                    "q_reward_type": q_reward_type,
                    "enable_q_loss": q_mode == "training",
                    "lambda_q_ce": 1.0,
                    "use_temporal_refine": refine,
                    "use_mask_memory": memory_mode == "global" and use_dynakey,
                    "use_memory_readout": memory_mode == "global" and use_dynakey,
                    "temporal_residual_init_scale": 0.1,
                    "temporal_gate_bias": -2.0,
                    "spatial_memory_slots": 3,
                    "spatial_memory_size": 8,
                    "spatial_memory_confidence_threshold": 0.0,
                    "spatial_memory_fg_ratio_min": 0.0,
                    "spatial_memory_fg_ratio_max": 1.0,
                    "mask_memory_confidence_threshold": 0.0,
                    "mask_memory_fg_ratio_min": 0.0,
                    "mask_memory_fg_ratio_max": 1.0,
                },
            }
        }
    )


def _batch(batch_size=2, frames=3, height=32, width=40, init_mode="pred_or_zero"):
    rgb = torch.randn(batch_size, frames, 1, height, width)
    cls_gt = torch.zeros(batch_size, frames, 1, height, width, dtype=torch.long)
    cls_gt[..., height // 4 : height // 2, width // 4 : width // 2] = 1
    ff_gt = F.one_hot(cls_gt[:, :1, 0], num_classes=2)[..., 1:].permute(0, 1, 4, 2, 3).float()
    return {
        "rgb": rgb,
        "ff_gt": ff_gt,
        "cls_gt": cls_gt,
        "label_valid": torch.ones(batch_size, frames, dtype=torch.bool),
        "supervised_indices": torch.ones(batch_size, frames, dtype=torch.bool),
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
        "init_mode": init_mode,
        "current_iter": 2,
    }


class SpatialDynaKeyTests(unittest.TestCase):
    def test_spatial_memory_state_retrieval_and_update_shapes(self):
        mem = SpatialDynaKeyMemory(6, num_slots=2, spatial_size=5, confidence_threshold=0.0, fg_ratio_min=0.0, fg_ratio_max=1.0)
        value = torch.randn(2, 1, 6, 7, 9)
        mask = torch.rand(2, 1, 28, 36)
        key = torch.randn(2, 6, 3, 4)
        pix = torch.randn(2, 6, 7, 9)
        read0 = mem.read(value, mask, key_BCHW=key, pixfeat_BCHW=pix, frame_index=1, total_frames=4)
        self.assertEqual(read0.feature.shape, (2, 1, 6, 5, 5))
        self.assertEqual(read0.delta.shape, (2, 1, 6, 5, 5))
        self.assertEqual(read0.gate.shape, (2, 1, 1, 5, 5))
        self.assertEqual(read0.mask_prior.shape, (2, 1, 5, 5))
        aux = mem.update(value, mask, key_BCHW=key, frame_index=1, total_frames=4)
        self.assertGreaterEqual(float(aux["spatial_memory_update_rate"]), 0.0)
        read1 = mem.read(value, mask, key_BCHW=key, pixfeat_BCHW=pix, frame_index=2, total_frames=4)
        self.assertEqual(read1.weights.shape, (2, 1, 2))
        self.assertTrue(torch.isfinite(read1.feature).all())
        self.assertGreater(int(mem._valid.sum().item()), 0)
        self.assertEqual(float(read1.aux["key_BCHW_used"].item()), 1.0)
        self.assertEqual(float(read1.aux["pixfeat_BCHW_used"].item()), 1.0)

    def test_segmentation_gain_reward_sign(self):
        gt = torch.ones(2, 1, 8, 8)
        before = torch.full_like(gt, -2.0)
        better = torch.full_like(gt, 2.0)
        worse = torch.full_like(gt, -4.0)
        self.assertTrue((segmentation_gain_reward(before, better, gt) > 0).all())
        self.assertTrue((segmentation_gain_reward(before, worse, gt) < 0).all())
        self.assertIsNone(segmentation_gain_reward(before, better, None))

    def test_config_switches_forward_paths(self):
        cases = [
            ("off", _cfg(use_dynakey=False, memory_mode="global", refine=False)),
            ("global", _cfg(memory_mode="global", refine=True)),
            ("spatial", _cfg(memory_mode="spatial", refine=False)),
            ("spatial_refine", _cfg(memory_mode="spatial", refine=True)),
            ("mid_fusion", _cfg(memory_mode="spatial", refine=False, memory_fusion_level="decoder", prediction_mode="memory_residual")),
            ("memory_primary", _cfg(memory_mode="spatial", refine=False, memory_fusion_level="decoder", prediction_mode="memory_primary", base_logits_weight=0.0)),
            ("memory_only", _cfg(memory_mode="spatial", refine=False, memory_fusion_level="decoder", prediction_mode="memory_primary", base_logits_weight=0.0, memory_only=True, lambda_memory_only=0.2)),
            ("base_detached", _cfg(memory_mode="spatial", refine=True, detach_base_logits=True, base_prob_dropout=0.2)),
            ("freeze_unext", _cfg(memory_mode="spatial", refine=True, freeze_unext=True)),
            ("q_diag", _cfg(memory_mode="spatial", refine=True, q_mode="diagnostic")),
            ("q_train", _cfg(memory_mode="spatial", refine=True, q_mode="training")),
        ]
        for name, cfg in cases:
            with self.subTest(name=name):
                model = UNeXtDynaKeySegmenter(cfg.model)
                out = model(_batch())
                self.assertEqual(out["logits_0"].shape, (2, 2, 32, 40))
                self.assertTrue(torch.isfinite(out["logits_2"]).all())
                aux = out["memory_aux_1"]
                if name == "off":
                    self.assertFalse(aux["dynakey_enabled"])
                    self.assertNotIn("spatial_q_values", aux)
                if name.startswith("spatial") or name.startswith("q_"):
                    self.assertTrue(aux["spatial_memory_enabled"])
                    self.assertIn("spatial_memory_valid_slots", aux)
                    self.assertEqual(float(aux["key_BCHW_used"].item()), 1.0)
                    self.assertEqual(float(aux["pixfeat_BCHW_used"].item()), 1.0)
                if name in {"mid_fusion", "memory_primary", "memory_only", "base_detached", "freeze_unext"}:
                    self.assertTrue(aux["use_mid_memory_fusion"] or name in {"base_detached", "freeze_unext"})
                if name == "q_diag":
                    self.assertIn("spatial_q_values", aux)
                    self.assertNotIn("spatial_q_target_action", aux)
                if name == "q_train":
                    self.assertIn("spatial_q_values", aux)
                    self.assertIn("spatial_q_target_action", aux)

    def test_spatial_q_training_backward(self):
        cfg = _cfg(memory_mode="spatial", refine=True, q_mode="training")
        model = UNeXtDynaKeySegmenter(cfg.model)
        data = _batch(batch_size=2, frames=3)
        out = model(data)
        data.update(out)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 32,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        self.assertIn("spatial_q_total", losses)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        self.assertIsNotNone(model.q_policy_head[0].weight.grad)
        self.assertIsNotNone(model.spatial_memory_proj.weight.grad)

    def test_spatial_without_q_has_no_q_loss(self):
        cfg = _cfg(memory_mode="spatial", refine=True, q_mode="diagnostic")
        model = UNeXtDynaKeySegmenter(cfg.model)
        data = _batch()
        data.update(model(data))
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 32,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        self.assertNotIn("spatial_q_total", losses)

    def test_q_training_latent_reward_without_gt_does_not_crash(self):
        cfg = _cfg(memory_mode="spatial", refine=True, q_mode="training", q_reward_type="latent")
        model = UNeXtDynaKeySegmenter(cfg.model)
        data = _batch()
        data.pop("cls_gt")
        data.pop("label_valid")
        out = model(data)
        aux = out["memory_aux_1"]
        self.assertIn("spatial_q_values", aux)
        self.assertIn("spatial_q_target_action", aux)
        self.assertEqual(aux["spatial_q_reward_type"], "latent")

    def test_phase_retrieval_can_change_top_slot(self):
        mem = SpatialDynaKeyMemory(
            2,
            num_slots=2,
            spatial_size=4,
            temperature=1.0,
            phase_weight=10.0,
            spatial_weight=0.0,
            shape_weight=0.0,
            confidence_threshold=0.0,
            fg_ratio_min=0.0,
            fg_ratio_max=1.0,
        )
        mem.reset_state(1, 1, torch.device("cpu"), torch.float32)
        mem._valid[:] = True
        mem._global[:] = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        mem._spatial[:] = 1.0
        mem._phase[0, 0, 0] = torch.tensor([0.1, 0.0, 0.0, 1.0])
        mem._phase[0, 0, 1] = torch.tensor([0.8, 0.0, 0.0, 1.0])
        value = torch.ones(1, 1, 2, 4, 4)
        key = torch.ones(1, 2, 4, 4)
        pix = torch.randn(1, 2, 4, 4)
        small = torch.full((1, 1, 16, 16), 0.1)
        large = torch.full((1, 1, 16, 16), 0.8)
        top_no_phase = mem.read(value, large, key_BCHW=key, pixfeat_BCHW=pix, use_phase=False).aux["spatial_memory_top_slot"]
        top_small = mem.read(value, small, key_BCHW=key, pixfeat_BCHW=pix, use_phase=True).aux["spatial_memory_top_slot"]
        top_large = mem.read(value, large, key_BCHW=key, pixfeat_BCHW=pix, use_phase=True).aux["spatial_memory_top_slot"]
        self.assertEqual(int(top_no_phase.item()), 0)
        self.assertEqual(int(top_small.item()), 0)
        self.assertEqual(int(top_large.item()), 1)

    def test_spatial_readout_is_not_broadcast_and_can_be_disabled(self):
        mem = SpatialDynaKeyMemory(3, num_slots=1, spatial_size=6, readout_scale=1.0, confidence_threshold=0.0, fg_ratio_min=0.0, fg_ratio_max=1.0)
        value = torch.randn(1, 1, 3, 6, 6)
        mask = torch.zeros(1, 1, 24, 24)
        mask[..., 4:18, 6:20] = 1.0
        key = torch.randn(1, 3, 6, 6)
        pix = torch.randn(1, 3, 6, 6)
        mem.update(value, mask, key_BCHW=key)
        read = mem.read(value + 0.5, mask, key_BCHW=key + 0.1, pixfeat_BCHW=pix, use_spatial_readout=True)
        self.assertGreater(float(read.aux["spatial_delta_hw_std"].mean()), 0.0)
        self.assertGreater(float(read.aux["spatial_gate_std"]), 0.0)
        self.assertGreater(float((read.feature - pix.unsqueeze(1)).abs().mean()), 0.0)
        disabled = mem.read(value + 0.5, mask, key_BCHW=key + 0.1, pixfeat_BCHW=pix, use_spatial_readout=False)
        self.assertTrue(torch.allclose(disabled.feature, pix.unsqueeze(1), atol=1e-6))
        self.assertEqual(float(disabled.aux["spatial_delta_norm"].max()), 0.0)

    def test_mid_level_memory_fusion_changes_logits_without_late_refine(self):
        torch.manual_seed(3)
        base_cfg = _cfg(memory_mode="spatial", refine=False, memory_fusion_level="late")
        mid_cfg = _cfg(
            memory_mode="spatial",
            refine=False,
            memory_fusion_level="decoder",
            prediction_mode="memory_residual",
            base_logits_weight=0.0,
        )
        base_model = UNeXtDynaKeySegmenter(base_cfg.model)
        mid_model = UNeXtDynaKeySegmenter(mid_cfg.model)
        mid_model.backbone.load_state_dict(base_model.backbone.state_dict())
        data = _batch(batch_size=2, frames=3)
        base_out = base_model(data)
        mid_out = mid_model(data)
        diff = (mid_out["logits_2"] - base_out["logits_2"]).abs().mean()
        self.assertGreater(float(diff), 1.0e-5)
        aux = mid_out["memory_aux_2"]
        self.assertTrue(aux["use_mid_memory_fusion"])
        self.assertIn("mid_memory_gate_mean", aux)
        self.assertGreater(float(aux["mid_memory_contribution_hw_std"].mean()), 0.0)
        self.assertGreater(float(aux["enhanced_feature_diff_norm"].mean()), 0.0)

    def test_memory_only_loss_backward(self):
        cfg = _cfg(
            memory_mode="spatial",
            refine=False,
            memory_fusion_level="decoder",
            prediction_mode="memory_primary",
            base_logits_weight=0.0,
            memory_only=True,
            lambda_memory_only=0.25,
        )
        model = UNeXtDynaKeySegmenter(cfg.model)
        data = _batch(batch_size=2, frames=3)
        out = model(data)
        self.assertIn("memory_only_logits", out["aux_1"])
        data.update(out)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 32,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        self.assertIn("aux_memory_only_ce", losses)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        self.assertIsNotNone(model.memory_object_head.weight.grad)
        self.assertIsNotNone(model.mid_memory_fusion.memory_proj.weight.grad)

    def test_shortcut_controls_do_not_break_training(self):
        cfg = _cfg(
            memory_mode="spatial",
            refine=True,
            memory_fusion_level="decoder",
            prediction_mode="memory_primary",
            base_logits_weight=0.0,
            detach_base_logits=True,
            base_prob_dropout=0.5,
            decoder_feature_dropout=0.2,
            q_mode="diagnostic",
        )
        model = UNeXtDynaKeySegmenter(cfg.model)
        model.train()
        data = _batch(batch_size=2, frames=3)
        out = model(data)
        self.assertTrue(torch.isfinite(out["logits_2"]).all())
        aux = out["memory_aux_2"]
        self.assertEqual(aux["prediction_mode"], "memory_primary")
        self.assertEqual(float(aux["base_logits_weight"].item()), 0.0)
        self.assertTrue(aux["detach_base_logits"])
        data.update(out)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 32,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        loss = LossComputer(cfg, stage_cfg).compute(data, [1, 1])["total_loss"]
        loss.backward()
        self.assertIsNotNone(model.mid_memory_fusion.memory_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
