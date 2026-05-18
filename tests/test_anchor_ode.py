import unittest
import tempfile
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dataset.echo import _apply_intensity_augmentation
from model.anchor_ode import UNeXtAnchorODEAffineSegmenter, UNeXtAnchorODESegmenter
from model.losses import LossComputer
from model.trainer import ModelEMA, Trainer
from model.utils.parameter_groups import get_parameter_groups
from train import resolve_mlflow_experiment_name


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "anchor_ode",
                "aux_loss": {
                    "sensory": {"weight": 0.0},
                    "query": {"weight": 0.0},
                },
                "memory_core": {"type": "none", "dynakey": {}},
                "temporal_memory": {"type": "none", "bpm": {}},
                "anchor_ode": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "base_dim": 8,
                    "value_dim": 16,
                    "num_slots": 4,
                    "state_dim": 24,
                    "hidden_dim": 24,
                    "anchor_size": 16,
                    "condition_dim": 12,
                    "ode_steps": 1,
                    "ode_gamma": 0.1,
                    "lambda_prior": 0.2,
                    "lambda_multiscale_prior": 0.02,
                    "lambda_geo": 0.05,
                    "lambda_temp_geo": 0.02,
                    "lambda_conf": 0.02,
                    "lambda_slot_balance": 0.001,
                },
            }
        }
    )


def _cfg_v2():
    cfg = _cfg()
    cfg.model.name = "anchor_ode_v2"
    cfg.model.anchor_ode.mode = "current_anchor_affine"
    cfg.model.anchor_ode.pop("anchor_size")
    cfg.model.anchor_ode.prior_residual_clip = 2.0
    cfg.model.anchor_ode.affine_max_translate = 0.12
    cfg.model.anchor_ode.affine_max_scale = 0.10
    cfg.model.anchor_ode.affine_max_rotate = 0.15
    cfg.model.anchor_ode.gate_warmup_iters = 0
    cfg.model.anchor_ode.lambda_base_seg = 0.5
    cfg.model.anchor_ode.lambda_guided_seg = 0.2
    cfg.model.anchor_ode.lambda_warp_prior = 0.1
    cfg.model.anchor_ode.lambda_affine_reg = 0.02
    return cfg


def _cfg_v2_tuned():
    cfg = _cfg_v2()
    cfg.model.anchor_ode.gate_init_bias = -2.5
    cfg.model.anchor_ode.confidence_prior_bias = -1.0
    cfg.model.anchor_ode.confidence_base_bias = 1.0
    cfg.model.anchor_ode.confidence_update_bias = -0.5
    cfg.model.anchor_ode.confidence_boundary_bias = -0.5
    cfg.model.anchor_ode.confidence_scale_bias = -1.0
    cfg.model.anchor_ode.confidence_slot_bias = 0.0
    cfg.model.anchor_ode.gate_warmup_iters = 400
    cfg.model.anchor_ode.lambda_base_seg = 0.3
    cfg.model.anchor_ode.lambda_warp_prior = 0.05
    cfg.model.anchor_ode.lambda_conf = 0.03
    cfg.model.anchor_ode.lambda_slot_balance = 0.0003
    cfg.model.anchor_ode.lambda_affine_reg = 0.01
    return cfg


def _batch(batch_size=2, frames=3, height=32, width=40):
    rgb = torch.randn(batch_size, frames, 1, height, width)
    cls_gt = torch.randint(0, 2, (batch_size, frames, 1, height, width), dtype=torch.long)
    ff_gt = torch.zeros(batch_size, 1, 1, height, width)
    return {
        "rgb": rgb,
        "ff_gt": ff_gt,
        "cls_gt": cls_gt,
        "label_valid": torch.ones(batch_size, frames, dtype=torch.bool),
        "supervised_indices": torch.ones(batch_size, frames, dtype=torch.bool),
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
        "init_mode": "pred_or_zero",
    }


class AnchorODETests(unittest.TestCase):
    def test_forward_shapes_and_skip_affines(self):
        torch.manual_seed(11)
        model = UNeXtAnchorODESegmenter(_cfg().model)
        out = model(_batch())
        for ti in range(3):
            self.assertEqual(out[f"logits_{ti}"].shape, (2, 2, 32, 40))
            self.assertEqual(out[f"masks_{ti}"].shape, (2, 1, 32, 40))
            aux = out[f"memory_aux_{ti}"]["anchor_ode_aux"]
            self.assertIn("slot_weights", aux)
            self.assertIn("prior_logits", aux)
            self.assertIn("confidence", aux)
            for name in ("affine_low", "affine_mid", "affine_high", "affine_dec"):
                self.assertEqual(aux[name].shape, (2, 1, 6))
                self.assertTrue(torch.isfinite(aux[name]).all())
            self.assertTrue(torch.allclose(aux["slot_weights"].sum(dim=-1), torch.ones(2, 1), atol=1.0e-5))
            self.assertEqual(aux["warped_priors"]["low"].shape[-2:], (16, 20))
            self.assertEqual(aux["warped_priors"]["mid"].shape[-2:], (8, 10))
            self.assertEqual(aux["warped_priors"]["high"].shape[-2:], (4, 5))
            self.assertEqual(aux["warped_priors"]["dec"].shape[-2:], (32, 40))

    def test_loss_backward_has_expected_grads(self):
        torch.manual_seed(12)
        cfg = _cfg()
        model = UNeXtAnchorODESegmenter(cfg.model)
        data = _batch(batch_size=2, frames=3)
        out = model(data)
        data.update(out)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 64,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        self.assertIn("aux_anchor_ode_prior", losses)
        self.assertIn("aux_anchor_ode_geo", losses)
        loss = losses["total_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.backbone.input_down[0].weight.grad)
        self.assertIsNotNone(model.anchor_bank.anchors["dec"].grad)
        self.assertIsNotNone(model.affine_regressor.net[-1].weight.grad)
        self.assertIsNotNone(model.confidence.net[-1].weight.grad)


class AnchorODEV2Tests(unittest.TestCase):
    def test_tuned_initialization_keeps_old_defaults_available(self):
        old_model = UNeXtAnchorODEAffineSegmenter(_cfg_v2().model)
        tuned_model = UNeXtAnchorODEAffineSegmenter(_cfg_v2_tuned().model)
        self.assertTrue(torch.allclose(old_model.gate_head[-1].bias, torch.full_like(old_model.gate_head[-1].bias, -4.0)))
        self.assertTrue(torch.allclose(tuned_model.gate_head[-1].bias, torch.full_like(tuned_model.gate_head[-1].bias, -2.5)))
        conf_bias = tuned_model.confidence.net[-1].bias.detach()
        expected = torch.tensor([-1.0, 1.0, -0.5, -0.5, -1.0, -1.0, -1.0, -1.0, 0.0])
        self.assertTrue(torch.allclose(conf_bias.cpu(), expected, atol=1.0e-6))
        self.assertLess(float(tuned_model.ode_bank.selector[-1].weight.detach().abs().max()), 0.01)

    def test_current_anchor_affine_forward_contract(self):
        torch.manual_seed(21)
        model = UNeXtAnchorODEAffineSegmenter(_cfg_v2().model)
        out = model(_batch())
        for ti in range(3):
            self.assertEqual(out[f"logits_{ti}"].shape, (2, 2, 32, 40))
            self.assertEqual(out[f"masks_{ti}"].shape, (2, 1, 32, 40))
            aux = out[f"memory_aux_{ti}"]["anchor_ode_aux"]
            self.assertEqual(aux["mode"], "current_anchor_affine")
            for key in (
                "base_object_logits",
                "guided_object_logits",
                "final_object_logits",
                "warped_anchor_low",
                "warped_anchor_mid",
                "warped_anchor_high",
                "warped_anchor_dec",
                "base_geometry",
                "final_geometry",
                "confidence_scale",
                "confidence_slot",
                "effective_slot_confidence",
                "slot_confidence",
                "geometry_delta",
                "selected_motion_embed",
            ):
                self.assertIn(key, aux)
            self.assertTrue(torch.allclose(aux["slot_weights"].sum(dim=-1), torch.ones(2, 1), atol=1.0e-5))
            self.assertEqual(aux["warped_anchor_low"].shape[-2:], (16, 20))
            self.assertEqual(aux["warped_anchor_mid"].shape[-2:], (8, 10))
            self.assertEqual(aux["warped_anchor_high"].shape[-2:], (4, 5))
            self.assertEqual(aux["warped_anchor_dec"].shape[-2:], (32, 40))
            for name in ("affine_low", "affine_mid", "affine_high", "affine_dec"):
                affine = aux[name]
                self.assertTrue(torch.isfinite(affine).all())
                self.assertLessEqual(float(affine[..., 0:2].abs().max()), 0.1201)
                self.assertLessEqual(float((affine[..., 2:4] - 1.0).abs().max()), 0.1001)
                self.assertLessEqual(float(affine[..., 4:6].abs().max()), 0.1501)
            self.assertEqual(aux["confidence_scale"].shape, (2, 1, 4))
            self.assertEqual(aux["confidence_boundary"].shape, (2, 1))

    def test_identity_init_stays_close_to_base(self):
        torch.manual_seed(22)
        model = UNeXtAnchorODEAffineSegmenter(_cfg_v2().model)
        out = model(_batch(batch_size=1, frames=1))
        aux = out["memory_aux_0"]["anchor_ode_aux"]
        diff = (aux["final_object_logits"] - aux["base_object_logits"]).abs().mean()
        self.assertLess(float(diff), 0.05)

    def test_first_frame_history_bootstrap_removes_zero_history_velocity(self):
        torch.manual_seed(24)
        model = UNeXtAnchorODEAffineSegmenter(_cfg_v2_tuned().model)
        out = model(_batch(batch_size=1, frames=1))
        aux = out["memory_aux_0"]["anchor_ode_aux"]
        self.assertLess(float(aux["base_geometry"][..., 8:11].abs().max()), 1.0e-6)

    def test_prior_influence_fusion_extremes(self):
        model = UNeXtAnchorODEAffineSegmenter(_cfg_v2().model)
        base = torch.zeros(2, 1, 4, 5)
        guided = torch.full_like(base, 0.75)
        zero_conf = torch.zeros(2, 1)
        one_conf = torch.ones(2, 1)
        self.assertTrue(torch.allclose(model._fuse_logits(base, guided, zero_conf), base))
        self.assertTrue(torch.allclose(model._fuse_logits(base, guided, one_conf), guided))

    def test_confidence_update_gates_history(self):
        model = UNeXtAnchorODEAffineSegmenter(_cfg_v2().model)
        prev = model._empty_prev(1, 1, torch.device("cpu"), torch.float32)
        prev["valid"].fill_(1.0)
        candidate = {k: torch.ones_like(v) for k, v in prev.items()}
        candidate["affine"][..., 2:4] = 1.5
        out_zero = model._gated_history_update(prev, candidate, torch.zeros(1, 1))
        out_one = model._gated_history_update(prev, candidate, torch.ones(1, 1))
        self.assertTrue(torch.allclose(out_zero["base_geometry"], prev["base_geometry"]))
        self.assertTrue(torch.allclose(out_zero["slot_weights"], prev["slot_weights"]))
        self.assertTrue(torch.allclose(out_one["base_geometry"], candidate["base_geometry"]))
        self.assertTrue(torch.allclose(out_one["affine"], candidate["affine"]))

        invalid_prev = model._empty_prev(1, 1, torch.device("cpu"), torch.float32)
        out_invalid = model._gated_history_update(invalid_prev, candidate, torch.zeros(1, 1))
        self.assertTrue(torch.allclose(out_invalid["base_geometry"], candidate["base_geometry"]))

    def test_v2_loss_backward_has_expected_grads(self):
        torch.manual_seed(23)
        cfg = _cfg_v2()
        model = UNeXtAnchorODEAffineSegmenter(cfg.model)
        data = _batch(batch_size=2, frames=3)
        out = model(data)
        data.update(out)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 64,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        self.assertIn("aux_anchor_ode_base", losses)
        self.assertIn("aux_anchor_ode_prior", losses)
        self.assertIn("aux_anchor_ode_guided", losses)
        self.assertIn("aux_anchor_ode_affine_reg", losses)
        loss = losses["total_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.backbone.input_down[0].weight.grad)
        self.assertIsNotNone(model.ode_bank.affine_velocity.grad)
        self.assertIsNotNone(model.affine_regressor.net[-1].weight.grad)
        self.assertIsNotNone(model.gate_head[-1].weight.grad)
        self.assertIsNotNone(model.confidence.net[-1].weight.grad)

    def test_v2_guidance_modes_forward(self):
        for mode, channels in (("warp_delta_boundary", 3), ("warp_delta", 2), ("warp_only", 1)):
            with self.subTest(mode=mode):
                cfg = _cfg_v2()
                cfg.model.anchor_ode.guidance_input_mode = mode
                model = UNeXtAnchorODEAffineSegmenter(cfg.model)
                out = model(_batch(batch_size=1, frames=1))
                aux = out["memory_aux_0"]["anchor_ode_aux"]
                self.assertEqual(aux["guidance_input_mode"], mode)
                first_proj = model.guidance_projs["low"]
                first_conv = first_proj[0] if isinstance(first_proj, torch.nn.Sequential) else first_proj
                self.assertEqual(first_conv.in_channels, channels)
                self.assertEqual(out["logits_0"].shape, (1, 2, 32, 40))

    def test_v2_guidance_hidden_concat_and_decoder_disable_backward(self):
        torch.manual_seed(25)
        cfg = _cfg_v2()
        cfg.model.anchor_ode.guidance_proj_hidden_dim = 16
        cfg.model.anchor_ode.guidance_fusion_mode = "residual_concat"
        cfg.model.anchor_ode.decoder_guidance_enabled = False
        model = UNeXtAnchorODEAffineSegmenter(cfg.model)
        data = _batch(batch_size=2, frames=2)
        out = model(data)
        aux = out["memory_aux_0"]["anchor_ode_aux"]
        self.assertEqual(aux["guidance_fusion_mode"], "residual_concat")
        self.assertEqual(float(aux["decoder_guidance_enabled"].item()), 0.0)
        self.assertIsInstance(model.guidance_projs["low"], torch.nn.Sequential)
        self.assertIn("low", model.guidance_concat_projs)
        data.update(out)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 64,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        loss = LossComputer(cfg, stage_cfg).compute(data, [1, 1])["total_loss"]
        loss.backward()
        self.assertIsNotNone(model.guidance_projs["low"][0].weight.grad)
        self.assertIsNotNone(model.guidance_concat_projs["low"][0].weight.grad)

    def test_eval_postprocess_largest_component_and_holes(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = OmegaConf.create(
            {
                "model": {"name": "anchor_ode_v2"},
                "evaluation": {"postprocess": {"enabled": True, "min_size": 4}},
            }
        )
        mask = torch.zeros(1, 1, 12, 12)
        mask[..., 2:8, 2:8] = 1.0
        mask[..., 4:6, 4:6] = 0.0
        mask[..., 10, 10] = 1.0
        processed = trainer._postprocess_binary_mask(mask)
        self.assertEqual(float(processed[..., 10, 10].item()), 0.0)
        self.assertEqual(float(processed[..., 4:6, 4:6].sum().item()), 4.0)

        trainer.cfg.evaluation.postprocess.enabled = False
        unchanged = trainer._postprocess_binary_mask(mask)
        self.assertTrue(torch.equal(unchanged, mask))

    def test_eval_postprocess_flags_and_min_size(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = OmegaConf.create(
            {
                "model": {"name": "anchor_ode_v2"},
                "evaluation": {
                    "postprocess": {
                        "enabled": True,
                        "largest_component": False,
                        "fill_holes": False,
                        "remove_small_objects": True,
                        "min_size": 4,
                        "binary_closing": False,
                    }
                },
            }
        )
        mask = torch.zeros(1, 1, 10, 10)
        mask[..., 1:4, 1:4] = 1.0
        mask[..., 2, 2] = 0.0
        mask[..., 7:9, 7:9] = 1.0
        mask[..., 9, 0] = 1.0
        processed = trainer._postprocess_binary_mask(mask)
        self.assertEqual(float(processed[..., 2, 2].item()), 0.0)
        self.assertEqual(float(processed[..., 7:9, 7:9].sum().item()), 4.0)
        self.assertEqual(float(processed[..., 9, 0].item()), 0.0)

    def test_ema_update_swap_restore_and_best_save(self):
        module = torch.nn.Conv2d(1, 1, kernel_size=1)
        trainer = object.__new__(Trainer)
        trainer.model = module
        trainer.ema = ModelEMA(module, decay=0.5)
        trainer.ema_enabled = True
        trainer.ema_eval = True
        trainer.main_process = True
        trainer.best_val_metric = -float("inf")
        trainer.best_val_threshold = 0.5
        trainer.cfg = OmegaConf.create(
            {
                "evaluation": {
                    "best_ckpt_metric": "best_threshold_dice_frame_mean",
                    "postprocess": {"enabled": False},
                    "tta": {"enabled": False},
                },
                "model": {"name": "anchor_ode_v2"},
            }
        )

        with torch.no_grad():
            raw_before = module.weight.detach().clone()
            module.weight.add_(2.0)
        trainer.ema.update(module)
        self.assertFalse(torch.allclose(trainer.ema.state["weight"], module.weight.detach()))

        raw_state = trainer._swap_to_ema_for_eval()
        self.assertTrue(torch.allclose(module.weight.detach(), trainer.ema.state["weight"]))
        trainer._restore_model_state(raw_state)
        self.assertTrue(torch.allclose(module.weight.detach(), raw_before + 2.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.run_path = Path(tmpdir)
            trainer.log = type("DummyLog", (), {"info": lambda *_args, **_kwargs: None})()
            trainer._save_best_if_needed(
                "val",
                {"best_threshold_dice_frame_mean": 0.9, "best_val_threshold": 0.53},
                epoch=1,
                it=200,
                raw_state=raw_state,
            )
            self.assertTrue((Path(tmpdir) / "raw_at_best_ema.pth").exists())
            self.assertTrue((Path(tmpdir) / "best_ema.pth").exists())
            self.assertTrue((Path(tmpdir) / "best_summary.json").exists())

    def test_eval_tta_hflip_forward_restores_shape(self):
        class ToyModel(torch.nn.Module):
            def forward(self, batch):
                prob = batch["rgb"][:, 0].mean(dim=1, keepdim=True)
                logits = torch.logit(prob.clamp(1.0e-4, 1.0 - 1.0e-4))
                return {"masks_0": torch.zeros_like(prob), "logits_0": logits}

        trainer = object.__new__(Trainer)
        trainer.model = ToyModel()
        trainer.cfg = OmegaConf.create(
            {
                "evaluation": {"tta": {"enabled": True, "modes": ["identity", "hflip"]}},
            }
        )
        batch = {"rgb": torch.rand(2, 1, 1, 8, 10), "ff_gt": torch.zeros(2, 1, 1, 8, 10)}
        out = trainer._forward_eval_with_tta(batch)
        self.assertEqual(out["masks_0"].shape, (2, 1, 8, 10))
        self.assertTrue(torch.allclose(out["masks_0"], batch["rgb"][:, 0], atol=1.0e-5))
        self.assertFalse(torch.allclose(out["masks_0"], torch.zeros_like(out["masks_0"])))

        trainer.cfg.evaluation.tta.modes = ["identity", "scale_0.95", "scale_1.05"]
        out_scale = trainer._forward_eval_with_tta(batch)
        self.assertEqual(out_scale["masks_0"].shape, (2, 1, 8, 10))

    def test_anchor_ode_parameter_groups_use_requested_lr_ratios(self):
        model = UNeXtAnchorODEAffineSegmenter(_cfg_v2().model)
        stage_cfg = OmegaConf.create(
            {
                "learning_rate": 1.0e-4,
                "weight_decay": 1.0e-3,
                "embed_weight_decay": 0.0,
                "backbone_lr_ratio": 0.1,
                "unext_lr_ratio": 0.5,
                "anchor_ode_lr_ratio": 1.5,
            }
        )
        groups = get_parameter_groups(model, stage_cfg)
        by_name = {group["name"]: group for group in groups}
        self.assertAlmostEqual(by_name["unext_base"]["lr"], 5.0e-5)
        self.assertAlmostEqual(by_name["anchor_ode"]["lr"], 1.5e-4)
        self.assertGreater(sum(p.numel() for p in by_name["unext_base"]["params"]), 0)
        self.assertGreater(sum(p.numel() for p in by_name["anchor_ode"]["params"]), 0)

    def test_light_echo_intensity_augmentation_is_train_only_safe(self):
        torch.manual_seed(31)
        frames = torch.full((3, 1, 8, 8), 0.5)
        cfg = {"enabled": True, "brightness": 0.04, "contrast": 0.08, "gamma": 0.08}
        augmented = _apply_intensity_augmentation(frames, cfg)
        self.assertEqual(augmented.shape, frames.shape)
        self.assertGreaterEqual(float(augmented.min()), 0.0)
        self.assertLessEqual(float(augmented.max()), 1.0)
        self.assertFalse(torch.equal(augmented, frames))

    def test_next_configs_compose_and_keep_old_tuned_defaults(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        with initialize_config_dir(config_dir=config_dir, version_base="1.3.2"):
            echo_cfg = compose(config_name="anchor_ode_v2_echo_more_guidance")
            camus_cfg = compose(config_name="anchor_ode_v2_camus_trust_guided")
            echo_off = compose(config_name="anchor_ode_v2_echo_more_guidance_no_postprocess")
            tuned_echo = compose(config_name="anchor_ode_v2_tuned_echo")

        self.assertEqual(echo_cfg.exp_id, "anchor_ode_v2_echo_more_guidance")
        self.assertEqual(camus_cfg.exp_id, "anchor_ode_v2_camus_trust_guided")
        self.assertEqual(echo_cfg.main_training.num_iterations, 4000)
        self.assertEqual(list(echo_cfg.main_training.lr_schedule_steps), [2000, 3200])
        self.assertAlmostEqual(echo_cfg.main_training.unext_lr_ratio, 0.5)
        self.assertAlmostEqual(echo_cfg.main_training.anchor_ode_lr_ratio, 1.5)
        self.assertAlmostEqual(echo_cfg.model.anchor_ode.gate_init_bias, -2.0)
        self.assertAlmostEqual(echo_cfg.model.anchor_ode.confidence_prior_bias, -0.75)
        self.assertAlmostEqual(echo_cfg.model.anchor_ode.affine_max_translate, 0.10)
        self.assertAlmostEqual(echo_cfg.model.anchor_ode.affine_max_rotate, 0.12)
        self.assertTrue(echo_cfg.augmentation.enabled)
        self.assertTrue(echo_cfg.evaluation.postprocess.enabled)
        self.assertFalse(echo_off.evaluation.postprocess.enabled)
        self.assertAlmostEqual(camus_cfg.model.anchor_ode.confidence_prior_bias, -0.5)
        self.assertFalse(camus_cfg.augmentation.enabled)
        self.assertAlmostEqual(tuned_echo.model.anchor_ode.gate_init_bias, -2.5)
        self.assertEqual(tuned_echo.main_training.num_iterations, 3000)

    def test_hparam_configs_compose(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        names = [
            "anchor_ode_v2_hparam_echo_e1",
            "anchor_ode_v2_hparam_echo_e2",
            "anchor_ode_v2_hparam_echo_e3",
            "anchor_ode_v2_hparam_camus_c1",
            "anchor_ode_v2_hparam_camus_c2",
            "anchor_ode_v2_hparam_camus_c3",
        ]
        with initialize_config_dir(config_dir=config_dir, version_base="1.3.2"):
            cfgs = {name: compose(config_name=name) for name in names}
        self.assertAlmostEqual(cfgs["anchor_ode_v2_hparam_echo_e1"].model.anchor_ode.gate_init_bias, -2.3)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_hparam_echo_e1"].main_training.anchor_ode_lr_ratio, 1.0)
        self.assertEqual(cfgs["anchor_ode_v2_hparam_echo_e2"].main_training.num_iterations, 4000)
        self.assertEqual(list(cfgs["anchor_ode_v2_hparam_echo_e2"].main_training.lr_schedule_steps), [2000, 3200])
        self.assertEqual(cfgs["anchor_ode_v2_hparam_echo_e3"].model.anchor_ode.guidance_input_mode, "warp_delta")
        self.assertAlmostEqual(cfgs["anchor_ode_v2_hparam_camus_c1"].model.anchor_ode.confidence_prior_bias, -0.75)
        self.assertEqual(cfgs["anchor_ode_v2_hparam_camus_c2"].model.anchor_ode.guidance_proj_hidden_dim, 16)
        self.assertFalse(cfgs["anchor_ode_v2_hparam_camus_c3"].model.anchor_ode.decoder_guidance_enabled)
        for cfg in cfgs.values():
            self.assertTrue(cfg.evaluation.postprocess.enabled)
            self.assertEqual(cfg.evaluation.protocol_version, "v3_current_anchor_affine")

    def test_brush_configs_compose_and_script_is_present(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        with initialize_config_dir(config_dir=config_dir, version_base="1.3.2"):
            echo_cfg = compose(config_name="anchor_ode_v2_brush_echo")
            camus_cfg = compose(config_name="anchor_ode_v2_brush_camus")

        self.assertEqual(echo_cfg.exp_id, "anchor_ode_v2_brush_echo")
        self.assertEqual(camus_cfg.exp_id, "anchor_ode_v2_brush_camus")
        for cfg in (echo_cfg, camus_cfg):
            self.assertTrue(cfg.main_training.use_ema)
            self.assertAlmostEqual(cfg.main_training.ema_decay, 0.999)
            self.assertEqual(cfg.main_training.ema_start_iter, 100)
            self.assertTrue(cfg.main_training.ema_eval)
            self.assertTrue(cfg.evaluation.tta.enabled)
            self.assertEqual(list(cfg.evaluation.tta.modes), ["identity", "hflip"])
            self.assertAlmostEqual(cfg.evaluation.threshold_search_start, 0.35)
            self.assertAlmostEqual(cfg.evaluation.threshold_search_end, 0.65)
            self.assertAlmostEqual(cfg.evaluation.threshold_search_step, 0.01)
            self.assertFalse(cfg.evaluation.postprocess.binary_closing)
        self.assertEqual(echo_cfg.evaluation.postprocess.min_size, 32)
        self.assertEqual(camus_cfg.evaluation.postprocess.min_size, 8)
        self.assertTrue((Path(__file__).resolve().parents[1] / "scripts/run_anchor_ode_v2_brush_camus_echo_tmux.sh").exists())

    def test_v3_configs_compose(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        names = [
            "anchor_ode_v2_v3_echo_e2_long_raw_tta",
            "anchor_ode_v2_v3_echo_boundaryless_long",
            "anchor_ode_v2_v3_echo_capacity_mild",
            "anchor_ode_v2_v3_camus_skip_only_tta",
            "anchor_ode_v2_v3_camus_skip_only_long",
            "anchor_ode_v2_v3_camus_skip_only_capacity",
        ]
        with initialize_config_dir(config_dir=config_dir, version_base="1.3.2"):
            cfgs = {name: compose(config_name=name) for name in names}

        for cfg in cfgs.values():
            self.assertFalse(cfg.main_training.use_ema)
            self.assertFalse(cfg.main_training.ema_eval)
            self.assertTrue(cfg.evaluation.tta.enabled)
            self.assertEqual(list(cfg.evaluation.tta.modes), ["identity", "hflip"])
            self.assertAlmostEqual(cfg.evaluation.threshold_search_start, 0.30)
            self.assertAlmostEqual(cfg.evaluation.threshold_search_end, 0.75)
            self.assertAlmostEqual(cfg.evaluation.threshold_search_step, 0.01)
            self.assertTrue(cfg.evaluation.postprocess.enabled)
            self.assertFalse(cfg.evaluation.postprocess.binary_closing)
            self.assertEqual(cfg.evaluation.protocol_version, "v3_current_anchor_affine")
            self.assertEqual(resolve_mlflow_experiment_name(cfg), "anchor_ode")

        for name in (
            "anchor_ode_v2_v3_echo_e2_long_raw_tta",
            "anchor_ode_v2_v3_echo_boundaryless_long",
            "anchor_ode_v2_v3_echo_capacity_mild",
            "anchor_ode_v2_v3_camus_skip_only_long",
        ):
            self.assertEqual(cfgs[name].main_training.num_iterations, 4000)
            self.assertEqual(list(cfgs[name].main_training.lr_schedule_steps), [2000, 3200])

        self.assertEqual(cfgs["anchor_ode_v2_v3_camus_skip_only_tta"].main_training.num_iterations, 3000)
        self.assertEqual(cfgs["anchor_ode_v2_v3_camus_skip_only_capacity"].main_training.num_iterations, 3000)
        self.assertEqual(cfgs["anchor_ode_v2_v3_echo_boundaryless_long"].model.anchor_ode.guidance_input_mode, "warp_delta")
        self.assertEqual(cfgs["anchor_ode_v2_v3_echo_capacity_mild"].model.anchor_ode.hidden_dim, 160)
        self.assertEqual(cfgs["anchor_ode_v2_v3_echo_capacity_mild"].model.anchor_ode.guidance_proj_hidden_dim, 16)
        self.assertFalse(cfgs["anchor_ode_v2_v3_camus_skip_only_tta"].model.anchor_ode.decoder_guidance_enabled)
        self.assertFalse(cfgs["anchor_ode_v2_v3_camus_skip_only_long"].model.anchor_ode.decoder_guidance_enabled)
        self.assertFalse(cfgs["anchor_ode_v2_v3_camus_skip_only_capacity"].model.anchor_ode.decoder_guidance_enabled)
        self.assertEqual(cfgs["anchor_ode_v2_v3_camus_skip_only_capacity"].model.anchor_ode.hidden_dim, 160)
        self.assertEqual(cfgs["anchor_ode_v2_v3_camus_skip_only_capacity"].model.anchor_ode.guidance_proj_hidden_dim, 16)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v3_camus_skip_only_capacity"].model.anchor_ode.lambda_guided_seg, 0.25)
        self.assertEqual(cfgs["anchor_ode_v2_v3_echo_e2_long_raw_tta"].evaluation.postprocess.min_size, 32)
        self.assertEqual(cfgs["anchor_ode_v2_v3_camus_skip_only_tta"].evaluation.postprocess.min_size, 8)
        self.assertTrue((Path(__file__).resolve().parents[1] / "scripts/run_anchor_ode_v2_v3_camus_echo_queue_tmux.sh").exists())

    def test_v4_configs_compose(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        echo_names = [
            "anchor_ode_v2_v4_echo_e2_raw_fine",
            "anchor_ode_v2_v4_echo_base_guard",
            "anchor_ode_v2_v4_echo_skip_only_raw",
        ]
        camus_names = [
            "anchor_ode_v2_v4_camus_skip_long_repro",
            "anchor_ode_v2_v4_camus_early_sched",
            "anchor_ode_v2_v4_camus_trust_guided",
        ]
        with initialize_config_dir(config_dir=config_dir, version_base="1.3.2"):
            cfgs = {name: compose(config_name=name) for name in echo_names + camus_names}

        for cfg in cfgs.values():
            self.assertFalse(cfg.main_training.use_ema)
            self.assertFalse(cfg.main_training.ema_eval)
            self.assertAlmostEqual(cfg.evaluation.threshold_search_start, 0.30)
            self.assertAlmostEqual(cfg.evaluation.threshold_search_end, 0.75)
            self.assertAlmostEqual(cfg.evaluation.threshold_search_step, 0.01)
            self.assertTrue(cfg.evaluation.postprocess.enabled)
            self.assertEqual(cfg.evaluation.protocol_version, "v3_current_anchor_affine")
            self.assertEqual(resolve_mlflow_experiment_name(cfg), "anchor_ode")

        for name in echo_names:
            cfg = cfgs[name]
            self.assertEqual(cfg.main_training.num_iterations, 4000)
            self.assertEqual(list(cfg.main_training.lr_schedule_steps), [2000, 3200])
            self.assertFalse(cfg.evaluation.tta.enabled)
            self.assertEqual(list(cfg.evaluation.tta.modes), ["identity"])
            self.assertEqual(cfg.evaluation.postprocess.min_size, 16)
            self.assertTrue(cfg.evaluation.postprocess.binary_closing)

        for name in camus_names:
            cfg = cfgs[name]
            self.assertTrue(cfg.evaluation.tta.enabled)
            self.assertEqual(list(cfg.evaluation.tta.modes), ["identity", "hflip"])
            self.assertEqual(cfg.evaluation.postprocess.min_size, 8)
            self.assertFalse(cfg.evaluation.postprocess.binary_closing)
            self.assertFalse(cfg.model.anchor_ode.decoder_guidance_enabled)

        self.assertEqual(cfgs["anchor_ode_v2_v4_camus_early_sched"].main_training.num_iterations, 3000)
        self.assertEqual(list(cfgs["anchor_ode_v2_v4_camus_early_sched"].main_training.lr_schedule_steps), [1600, 2400])
        self.assertEqual(cfgs["anchor_ode_v2_v4_camus_skip_long_repro"].main_training.num_iterations, 4000)
        self.assertEqual(list(cfgs["anchor_ode_v2_v4_camus_skip_long_repro"].main_training.lr_schedule_steps), [2000, 3200])
        self.assertFalse(cfgs["anchor_ode_v2_v4_echo_skip_only_raw"].model.anchor_ode.decoder_guidance_enabled)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_echo_base_guard"].model.anchor_ode.gate_init_bias, -2.7)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_echo_base_guard"].model.anchor_ode.confidence_prior_bias, -1.2)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_echo_base_guard"].model.anchor_ode.lambda_base_seg, 0.45)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_echo_base_guard"].model.anchor_ode.lambda_guided_seg, 0.15)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_echo_base_guard"].model.anchor_ode.prior_residual_clip, 1.0)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_echo_base_guard"].model.anchor_ode.lambda_affine_reg, 0.02)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_camus_trust_guided"].model.anchor_ode.confidence_prior_bias, -0.65)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_camus_trust_guided"].model.anchor_ode.lambda_conf, 0.015)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_camus_trust_guided"].model.anchor_ode.lambda_guided_seg, 0.22)
        self.assertAlmostEqual(cfgs["anchor_ode_v2_v4_camus_trust_guided"].model.anchor_ode.lambda_affine_reg, 0.008)
        self.assertTrue((Path(__file__).resolve().parents[1] / "scripts/run_anchor_ode_v2_v4_camus_echo_queue_tmux.sh").exists())


if __name__ == "__main__":
    unittest.main()
