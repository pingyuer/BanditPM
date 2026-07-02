import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.registry import build_model
from rebel.rebel import ReBelSegmenter


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "rebel",
                "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
                "rebel": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "belief_dim": 16,
                    "decoder_dim": 16,
                    "backbone": {"name": "official", "base_dim": 8, "mlp_expansion": 2.0, "latent_blocks": 1, "decoder_mlp_blocks": 1},
                    "ode": {"hidden_dim": 16, "num_blocks": 1, "max_offset_px_stage4": 4.0, "offset_warmup_iters": 10},
                    "memory": {"write_fast_init": 0.25, "write_slow_init": 0.05, "decay_fast_init": 0.75, "decay_slow_init": 0.95},
                    "correction": {"enabled": True, "init_scale": 0.25, "max_scale": 1.0, "start_iter": 1000},
                    "fusion": {"hidden_dim": 16, "min_base_weight": 0.05},
                },
            }
        }
    )


def test_rebel_forward_outputs_and_final_path_decoupled():
    model = build_model(_cfg(), device="cpu")
    data = {"rgb": torch.randn(2, 3, 1, 32, 32), "current_iter": 5}
    out = model(data)
    for key in ("logits", "base_logits", "belief_logits", "obs_logits", "correction_logits", "aux"):
        assert key in out
    assert out["logits"].shape == (2, 3, 2, 32, 32)
    assert out["base_logits"].shape == (2, 3, 2, 32, 32)
    assert not torch.allclose(out["logits"], out["base_logits"])
    assert "rebel/disagreement_mean" in out["aux"]
    assert "rebel/write_fast_mean" in out["aux"]
    assert "rebel/decay_slow_mean" in out["aux"]
    assert "rebel/arbitration_weight_base_mean" in out["aux"]
    assert "rebel/arbitration_entropy_mean" in out["aux"]


def test_rebel_registry_alias_builds():
    cfg = _cfg()
    cfg.model.name = "resampled_belief"
    assert build_model(cfg, device="cpu").__class__.__name__ == "ReBelSegmenter"


def test_rebel_belief_prior_logits_preserve_sigmoid_probability():
    model = build_model(_cfg(), device="cpu")
    z = torch.tensor([[[[0.0, 1.0, -1.0]]]])
    logits = model._belief_logits_from_prior(z, (1, 3))
    prob = torch.softmax(logits, dim=1)[:, 1]
    assert torch.allclose(prob, torch.sigmoid(z[:, 0]), atol=1e-6)


def test_rebel_correction_start_iter_gates_contribution():
    model = build_model(_cfg(), device="cpu")
    data = {"rgb": torch.randn(1, 2, 1, 32, 32), "current_iter": 0}
    out = model(data)
    assert out["correction_logits"].abs().max().item() == 0.0
    assert out["aux"]["rebel/correction_scale_mean"].item() == 0.0
    out_active = model({"rgb": data["rgb"], "current_iter": 1000})
    assert out_active["correction_logits"].abs().mean().item() > 0.0
    assert out_active["aux"]["rebel/correction_scale_mean"].item() > 0.0


def test_rebel_same_observation_different_memory_changes_final_logits():
    model = build_model(_cfg(), device="cpu")
    frame = torch.randn(1, 1, 1, 32, 32)
    data_a = {"rgb": torch.cat([torch.zeros_like(frame), frame], dim=1), "current_iter": 1000}
    data_b = {"rgb": torch.cat([torch.ones_like(frame), frame], dim=1), "current_iter": 1000}
    out_a = model(data_a)["logits_1"]
    out_b = model(data_b)["logits_1"]
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_rebel_same_memory_different_delta_mem_changes_final_logits():
    model = build_model(_cfg(), device="cpu")
    data = {"rgb": torch.randn(1, 2, 1, 32, 32), "current_iter": 1000}
    original_forward = model.ode.forward

    def with_delta(delta_x):
        def patched(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            out["delta_mem"] = torch.zeros_like(out["delta_mem"])
            out["delta_mem"][:, 0] = delta_x
            return out
        return patched

    model.ode.forward = with_delta(0.0)
    out_zero = model(data)["logits_1"]
    model.ode.forward = with_delta(1.0)
    out_shift = model(data)["logits_1"]
    assert not torch.allclose(out_zero, out_shift, atol=1e-5)


def test_rebel_changing_base_logits_after_init_can_drive_final_fallback():
    model = build_model(_cfg(), device="cpu")
    original_encoder = model.encoder
    frames = torch.randn(1, 2, 1, 32, 32)
    cached = [original_encoder(frames[:, 0]), original_encoder(frames[:, 1])]

    class FixedEncoder(nn.Module):
        def __init__(self, cached, delta):
            super().__init__()
            self.cached = cached
            self.delta = delta
            self.low_dim = original_encoder.low_dim
            self.mid_dim = original_encoder.mid_dim
            self.high_dim = original_encoder.high_dim
            self.i = 0

        def forward(self, _frame):
            item = {k: v.detach().clone() for k, v in self.cached[self.i].items()}
            if self.i == 1:
                item["base_logits"] = item["base_logits"] + self.delta
                item["logits"] = item["base_logits"]
            self.i += 1
            return item

    model.encoder = FixedEncoder(cached, 0.0)
    out_a = model({"rgb": frames, "current_iter": 1000})["logits_1"]
    model.encoder = FixedEncoder(cached, 10.0)
    out_b = model({"rgb": frames, "current_iter": 1000})["logits_1"]
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_rebel_decoder_disabled_cannot_produce_valid_final_logits():
    model = build_model(_cfg(), device="cpu")
    model.decoder = None
    try:
        model({"rgb": torch.randn(1, 1, 1, 32, 32), "current_iter": 1000})
        assert False, "ReBel without BeliefDecoder should not produce final logits"
    except TypeError:
        pass
