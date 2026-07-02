import torch

from rebel.memory import ResampledBeliefMemory


def test_rebel_memory_state_shapes_and_prior():
    mem = ResampledBeliefMemory(4)
    obs = torch.randn(2, 4, 6, 7)
    state = mem.init_state(obs)
    assert state["W_feat"].shape == (2, 4, 6, 7)
    assert state["S_feat"].shape == (2, 4, 6, 7)
    assert state["W_mask"].shape == (2, 1, 6, 7)
    assert state["S_mask"].shape == (2, 1, 6, 7)
    assert state["R"].shape == (2, 1, 6, 7)
    belief, prior = mem.read_memory(state)
    assert belief.shape == obs.shape
    assert prior.shape == (2, 1, 6, 7)


def test_rebel_memory_delta_update_and_reliability_gate():
    mem = ResampledBeliefMemory(1, stable_reliability_threshold=0.55, detach_update=True)
    old = torch.zeros(1, 1, 2, 2)
    state = {"W_feat": old.clone(), "S_feat": old.clone(), "W_mask": old.clone(), "S_mask": old.clone(), "R": old.clone()}
    new = torch.ones_like(old)
    write_fast = torch.full_like(old, 0.25)
    write_slow = torch.full_like(old, 0.05)
    decay_fast = torch.ones_like(old)
    decay_slow = torch.ones_like(old)
    out = mem.update(state, new, new, write_fast, write_slow, decay_fast, decay_slow, torch.zeros_like(old))
    assert torch.allclose(out["W_mask"], torch.full_like(old, 0.25))
    assert out["S_mask"].abs().max().item() == 0.0
    state["R"] = torch.ones_like(old)
    out = mem.update(state, new, new, write_fast, write_slow, decay_fast, decay_slow, torch.zeros_like(old))
    assert out["W_mask"].mean() > out["S_mask"].mean()


def test_rebel_memory_mask_decay_matches_feature_decay():
    mem = ResampledBeliefMemory(1, stable_reliability_threshold=0.0, detach_update=True)
    old = torch.zeros(1, 1, 2, 2)
    state = {"W_feat": old.clone(), "S_feat": old.clone(), "W_mask": old.clone(), "S_mask": old.clone(), "R": torch.ones_like(old)}
    new = torch.ones_like(old)
    write = torch.full_like(old, 0.25)
    decay = torch.full_like(old, 0.5)
    out = mem.update(state, new, new, write, write, decay, decay, torch.zeros_like(old))
    assert torch.allclose(out["W_feat"], torch.full_like(old, 0.125))
    assert torch.allclose(out["W_mask"], torch.full_like(old, 0.125))


def test_rebel_arbitration_memory_reliability_high_conflict_lowers_r_obs():
    mem = ResampledBeliefMemory(2)
    obs = torch.ones(1, 2, 3, 3)
    memory = torch.zeros_like(obs)
    obs_prob = torch.ones(1, 1, 3, 3)
    mem_prob = torch.zeros_like(obs_prob)
    r_raw = torch.full_like(obs_prob, 0.5)
    low_r = torch.zeros_like(obs_prob)
    high_r = torch.ones_like(obs_prob)
    _, r_obs_low_mem, _ = mem.arbitrate(obs, memory, mem_prob, obs_prob, low_r, r_raw)
    _, r_obs_high_mem, _ = mem.arbitrate(obs, memory, mem_prob, obs_prob, high_r, r_raw)
    assert r_obs_high_mem.mean() < r_obs_low_mem.mean()
