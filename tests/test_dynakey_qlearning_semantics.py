import unittest

import torch
from omegaconf import OmegaConf

from losses import LossComputer
from model.modules.dynakey import DynaKeyMemoryCore
from model.modules.dynakey.counterfactual import compute_counterfactual_returns
from model.modules.dynakey.ode_key_dictionary import ODEKeyDictionary
from model.modules.dynakey.q_maintainer import DynaKeyQMaintainer


def clone_state_tensors(dictionary):
    state = dictionary.clone_state()
    return {name: getattr(state, name) for name in ("center", "velocity", "scale", "age", "usage", "error_ema", "valid")}


def assert_batch_unchanged(testcase, before, dictionary, batch_idx):
    after = clone_state_tensors(dictionary)
    for name, tensor in before.items():
        testcase.assertTrue(torch.equal(tensor[batch_idx], after[name][batch_idx]), name)


def value_from_z(z):
    b, n, c = z.shape
    return z.view(b, n, c, 1, 1).expand(b, n, c, 2, 2).contiguous()


def make_loss_computer():
    cfg = OmegaConf.create(
        {
            "model": {
                "aux_loss": {"sensory": {"weight": 0.01}, "query": {"weight": 0.01}},
                "temporal_memory": {"bpm": {}},
                "memory_core": {
                    "dynakey": {
                        "ENABLE_Q_LOSS": True,
                        "LAMBDA_Q_CE": 1.0,
                        "LAMBDA_Q_ADV": 0.5,
                        "ADVANTAGE_CLAMP": 2.0,
                    }
                },
            }
        }
    )
    stage_cfg = OmegaConf.create(
        {
            "point_supervision": True,
            "train_num_points": 16,
            "oversample_ratio": 1.0,
            "importance_sample_ratio": 0.5,
        }
    )
    return LossComputer(cfg, stage_cfg)


class DynaKeyQLearningSemanticsTests(unittest.TestCase):
    def test_masked_actions_do_not_pollute_other_batch_items(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=3, ema_alpha=1.0)
        dictionary.reset_state(batch_size=2, num_objects=1, device=torch.device("cpu"))
        z0 = torch.tensor([[[0.0, 0.0]], [[10.0, 0.0]]])
        dictionary.spawn_masked(z0, torch.zeros_like(z0), enabled=torch.ones(2, 1, dtype=torch.bool))

        before = clone_state_tensors(dictionary)
        z1 = torch.tensor([[[1.0, 0.0]], [[11.0, 0.0]]])
        dictionary.spawn_masked(z1, z1 - z0, enabled=torch.tensor([[True], [False]]))
        self.assertEqual(dictionary.active_key_count()[0, 0].item(), 2)
        self.assertEqual(dictionary.active_key_count()[1, 0].item(), 1)
        assert_batch_unchanged(self, before, dictionary, 1)

        before = clone_state_tensors(dictionary)
        dictionary.update_masked(z0, z1, torch.zeros(2, 1, dtype=torch.long), enabled=torch.tensor([[True], [False]]))
        self.assertFalse(torch.equal(before["velocity"][0], dictionary.state.velocity[0]))
        assert_batch_unchanged(self, before, dictionary, 1)

        before = clone_state_tensors(dictionary)
        dictionary.delete_masked(torch.zeros(2, 1, dtype=torch.long), enabled=torch.tensor([[True], [True]]))
        self.assertEqual(dictionary.active_key_count()[0, 0].item(), 1)
        self.assertEqual(dictionary.active_key_count()[1, 0].item(), 1)
        assert_batch_unchanged(self, before, dictionary, 1)

    def test_counterfactual_initial_state_does_not_pollute_live_dictionary(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=3, ema_alpha=1.0)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        z0 = torch.tensor([[[0.0, 0.0]]])
        dictionary.spawn(z0, torch.zeros_like(z0))
        pre_action = dictionary.clone_state()
        dictionary.spawn(torch.tensor([[[5.0, 0.0]]]), torch.zeros_like(z0))
        live_before = clone_state_tensors(dictionary)

        _, aux = compute_counterfactual_returns(
            dictionary,
            z0,
            torch.tensor([[[1.0, 0.0]]]),
            initial_state=pre_action,
        )

        for name, tensor in live_before.items():
            self.assertTrue(torch.equal(tensor, clone_state_tensors(dictionary)[name]), name)
        self.assertEqual(aux["advantage_returns"].shape, (1, 1, 5))

    def test_q_target_never_selects_illegal_action(self):
        core = DynaKeyMemoryCore(
            {
                "BANK_SIZE": 1,
                "POLICY_MODE": "q_greedy",
                "ENABLE_Q_LOSS": True,
                "HIDDEN_DIM": 8,
                "DETACH_Q_STATE": True,
            },
            value_dim=2,
        )
        core.reset_state(1, 1, torch.device("cpu"))
        mask = torch.ones(1, 1, 2, 2)
        core(value_from_z(torch.tensor([[[0.0, 0.0]]])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        _, aux = core(value_from_z(torch.tensor([[[1.0, 0.0]]])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)

        target = aux["q_target_action"]
        action_mask = aux["action_mask"]
        self.assertIsNotNone(target)
        self.assertTrue(action_mask.gather(-1, target.unsqueeze(-1)).squeeze(-1).all())
        self.assertFalse(aux["invalid_q_targets"].any())

    def test_loss_skips_invalid_q_labels_without_nan(self):
        loss = make_loss_computer()
        q_values = torch.randn(2, 1, 5, requires_grad=True)
        data = {
            "rgb": torch.zeros(1, 2, 1, 4, 4),
            "memory_aux_1": {
                "dynakey_aux": {
                    "q_values": q_values,
                    "q_target_action": torch.tensor([[2], [4]]),
                    "advantage_returns": torch.randn(2, 1, 5),
                    "action_mask": torch.tensor(
                        [
                            [[True, True, False, False, False]],
                            [[True, True, True, True, False]],
                        ]
                    ),
                }
            },
        }
        terms = loss._compute_dynakey_q_loss(data)
        self.assertTrue(torch.isfinite(terms["dynakey_q_total"]))
        self.assertEqual(terms["dynakey_q_valid_samples"].item(), 0.0)
        terms["dynakey_q_total"].backward()
        self.assertIsNotNone(q_values.grad)

    def test_detach_q_state_keeps_q_head_trainable(self):
        core = DynaKeyMemoryCore(
            {
                "BANK_SIZE": 2,
                "POLICY_MODE": "q_greedy",
                "ENABLE_Q_LOSS": True,
                "HIDDEN_DIM": 8,
                "DETACH_Q_STATE": True,
            },
            value_dim=2,
        )
        core.reset_state(1, 1, torch.device("cpu"))
        mask = torch.ones(1, 1, 2, 2)
        core(value_from_z(torch.tensor([[[0.0, 0.0]]])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        _, aux = core(value_from_z(torch.tensor([[[1.0, 0.0]]])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)

        data = {"rgb": torch.zeros(1, 2, 1, 4, 4), "memory_aux_1": {"dynakey_aux": aux}}
        total = make_loss_computer()._compute_dynakey_q_loss(data)["dynakey_q_total"]
        total.backward()
        grads = [p.grad for p in core.q_maintainer.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
