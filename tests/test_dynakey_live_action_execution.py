import unittest

import torch

from model.modules.dynakey import DynaKeyMemoryCore


def make_core(policy_mode: str, forced_action: str | None = None, bank_size: int = 4):
    cfg = {
        "BANK_SIZE": bank_size,
        "DT": 1.0,
        "EMA_ALPHA": 1.0,
        "HIDDEN_DIM": 16,
        "POLICY_MODE": policy_mode,
        "RESIDUAL_SPAWN_THRESHOLD": 0.01,
        "SPLIT_EPS": 0.1,
        "SPLIT_SCALE_FACTOR": 0.7,
    }
    if forced_action is not None:
        cfg["FORCED_ACTION"] = forced_action
    core = DynaKeyMemoryCore(cfg, value_dim=2)
    core.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
    return core


def value_from_z(z):
    return z.view(1, 1, 2, 1, 1).expand(1, 1, 2, 2, 2).contiguous()


class DynaKeyLiveActionExecutionTests(unittest.TestCase):
    def test_forced_spawn_grows_active_key_count(self):
        core = make_core("forced", "spawn", bank_size=4)
        mask = torch.ones(1, 1, 2, 2)
        for z in (torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]), torch.tensor([3.0, 0.0])):
            _, aux = core(value_from_z(z), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        self.assertGreater(aux["active_key_count"].item(), 1)
        self.assertIn("executed_action", aux)
        self.assertGreater(aux["action_spawn"].item(), 0.0)

    def test_forced_split_increases_or_safely_reuses_slot(self):
        core = make_core("forced", "split", bank_size=2)
        mask = torch.ones(1, 1, 2, 2)
        _, _ = core(value_from_z(torch.tensor([0.0, 0.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        before = core.dictionary.active_key_count().clone()
        _, aux = core(value_from_z(torch.tensor([1.0, 0.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        self.assertGreaterEqual(aux["active_key_count"].item(), before.item())
        self.assertGreaterEqual(aux["active_key_count"].item(), 1)
        self.assertGreater(aux["action_split"].item(), 0.0)

    def test_forced_delete_cannot_empty_dictionary(self):
        core = make_core("forced", "delete", bank_size=3)
        mask = torch.ones(1, 1, 2, 2)
        _, aux = core(value_from_z(torch.tensor([0.0, 0.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        self.assertGreaterEqual(aux["active_key_count"].item(), 1)
        _, aux = core(value_from_z(torch.tensor([1.0, 0.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        self.assertGreaterEqual(aux["active_key_count"].item(), 1)

    def test_forced_update_changes_velocity(self):
        core = make_core("forced", "update", bank_size=3)
        mask = torch.ones(1, 1, 2, 2)
        _, _ = core(value_from_z(torch.tensor([0.0, 0.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        before = core.dictionary.state.velocity.clone()
        _, aux = core(value_from_z(torch.tensor([0.0, 1.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        self.assertFalse(torch.allclose(before, core.dictionary.state.velocity))
        self.assertGreater(aux["action_update"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
