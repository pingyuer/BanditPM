import torch

from rebel.ode_field import BeliefODEField


def test_rebel_ode_initializes_identity_offsets_and_gates():
    ode = BeliefODEField(8, hidden_dim=16, max_offset_px=6.0)
    x = torch.randn(2, 8, 4, 5)
    out = ode(x, x, x, torch.full((2, 1, 4, 5), 0.5), torch.full((2, 1, 4, 5), 0.5))
    assert torch.allclose(out["delta_obs"], torch.zeros_like(out["delta_obs"]), atol=1e-6)
    assert torch.allclose(out["delta_mem"], torch.zeros_like(out["delta_mem"]), atol=1e-6)
    assert abs(out["write_fast"].mean().item() - 0.25) < 1e-3
    assert abs(out["write_slow"].mean().item() - 0.05) < 1e-3
    assert abs(out["decay_fast"].mean().item() - 0.75) < 1e-3
    assert abs(out["decay_slow"].mean().item() - 0.95) < 1e-3
    assert "logits" not in out and "final_logits" not in out


def test_rebel_ode_offset_warmup_scale_reaches_full():
    ode = BeliefODEField(8, hidden_dim=16, max_offset_px=8.0, offset_warmup_iters=100, offset_warmup_start_ratio=0.25)
    assert abs(ode._offset_scale(0) - 2.0) < 1e-6
    assert abs(ode._offset_scale(100) - 8.0) < 1e-6
