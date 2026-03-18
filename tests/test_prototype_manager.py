from types import SimpleNamespace
import unittest

import torch

from model.modules.prototype_manager import BanditPrototypeManager


def build_cfg(**overrides):
    base = {
        "BANK_SIZE": 4,
        "PROTO_ALPHA": 0.1,
        "READOUT_TEMPERATURE": 1.0,
        "DEFAULT_ACTION": 1,
        "SPAWN_REPLACE_WHEN_FULL": True,
        "USE_RULE_BASED_POLICY": True,
        "USE_LEARNED_POLICY": False,
        "POLICY_LOSS_WEIGHT": 1.0,
        "SIM_THRESHOLD_HIGH": 0.8,
        "SIM_THRESHOLD_LOW": 0.5,
        "DEBUG_MODE": False,
        "PROTO_POOLING": "mask",
        "FUSION_TYPE": "add",
        "POLICY_HIDDEN_DIM": 64,
        "ACTION_COSTS": {"keep": 0.0, "refine": 0.05, "replace": 0.1, "spawn": 0.2},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class BanditPrototypeManagerTests(unittest.TestCase):
    def test_forward_returns_expected_shapes(self):
        manager = BanditPrototypeManager(build_cfg(), value_dim=256)
        value = torch.randn(2, 1, 256, 8, 8)
        feat = torch.randn(2, 256, 8, 8)
        mask = torch.rand(2, 1, 32, 32)

        manager.reset_state(batch_size=2, num_objects=1, device=value.device)
        conditioned, aux = manager(value, feat, mask)

        self.assertEqual(conditioned.shape, value.shape)
        self.assertEqual(aux["policy_logits"].shape, (2, 1, 4))
        self.assertEqual(aux["policy_labels"].shape, (2, 1))
        self.assertEqual(aux["policy_actions"].shape, (2, 1))
        self.assertEqual(aux["bank_proto"].shape, (2, 1, 4, 256))

    def test_rule_based_policy_spawns_on_empty_bank(self):
        manager = BanditPrototypeManager(build_cfg(SIM_THRESHOLD_HIGH=0.95, SIM_THRESHOLD_LOW=0.9), value_dim=256)
        value = torch.randn(1, 1, 256, 8, 8)
        feat = torch.randn(1, 256, 8, 8)
        mask = torch.rand(1, 1, 8, 8)

        manager.reset_state(batch_size=1, num_objects=1, device=value.device)
        _, aux = manager(value, feat, mask)

        self.assertTrue(torch.equal(aux["policy_actions"], torch.full((1, 1), 3, dtype=torch.long)))
        self.assertTrue(aux["bank_valid"].any())


if __name__ == "__main__":
    unittest.main()
