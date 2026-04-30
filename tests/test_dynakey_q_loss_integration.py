import unittest

import torch
from omegaconf import OmegaConf

from model.losses import LossComputer
from model.modules.dynakey import DynaKeyMemoryCore


def value_from_z(z):
    return z.view(1, 1, 2, 1, 1).expand(1, 1, 2, 2, 2).contiguous()


def make_cfg(enable=True):
    return OmegaConf.create(
        {
            "model": {
                "aux_loss": {
                    "sensory": {"weight": 0.01},
                    "query": {"weight": 0.01},
                },
                "temporal_memory": {"bpm": {}},
                "memory_core": {
                    "dynakey": {
                        "ENABLE_Q_LOSS": enable,
                        "LAMBDA_Q_CE": 0.7,
                        "LAMBDA_Q_ADV": 0.3,
                        "ADVANTAGE_CLAMP": 2.0,
                    }
                },
            }
        }
    )


class DynaKeyQLossIntegrationTests(unittest.TestCase):
    def test_live_path_emits_q_supervision_with_grad(self):
        core = DynaKeyMemoryCore(
            {
                "BANK_SIZE": 3,
                "POLICY_MODE": "q_greedy",
                "ENABLE_Q_LOSS": True,
                "HIDDEN_DIM": 8,
                "EMA_ALPHA": 1.0,
            },
            value_dim=2,
        )
        core.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        mask = torch.ones(1, 1, 2, 2)
        _, _ = core(value_from_z(torch.tensor([0.0, 0.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        _, aux = core(value_from_z(torch.tensor([1.0, 0.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)

        self.assertIn("q_values", aux)
        self.assertIn("q_target_action", aux)
        self.assertIn("advantage_returns", aux)
        self.assertTrue(aux["q_values"].requires_grad)
        self.assertEqual(aux["q_values"].shape, (1, 1, 5))

    def test_loss_computer_adds_dynakey_q_loss_when_enabled(self):
        cfg = make_cfg(enable=True)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": True,
                "train_num_points": 16,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        loss = LossComputer(cfg, stage_cfg)
        q_values = torch.randn(1, 1, 5, requires_grad=True)
        data = {
            "rgb": torch.zeros(1, 2, 1, 4, 4),
            "memory_aux_0": {"dynakey_aux": {}},
            "memory_aux_1": {
                "dynakey_aux": {
                    "q_values": q_values,
                    "q_target_action": torch.tensor([[2]]),
                    "advantage_returns": torch.tensor([[[0.0, -0.1, 1.0, -0.5, -0.2]]]),
                    "action_mask": torch.tensor([[[True, True, True, True, False]]]),
                }
            },
        }

        terms = loss._compute_dynakey_q_loss(data)

        self.assertIn("dynakey_q_ce", terms)
        self.assertIn("dynakey_q_adv", terms)
        self.assertGreater(terms["dynakey_q_total"].item(), 0.0)
        terms["dynakey_q_total"].backward()
        self.assertIsNotNone(q_values.grad)

    def test_loss_computer_ignores_dynakey_q_loss_when_disabled(self):
        cfg = make_cfg(enable=False)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": True,
                "train_num_points": 16,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        loss = LossComputer(cfg, stage_cfg)
        data = {
            "rgb": torch.zeros(1, 2, 1, 4, 4),
            "memory_aux_1": {
                "dynakey_aux": {
                    "q_values": torch.randn(1, 1, 5, requires_grad=True),
                    "q_target_action": torch.tensor([[2]]),
                    "advantage_returns": torch.randn(1, 1, 5),
                    "action_mask": torch.ones(1, 1, 5, dtype=torch.bool),
                }
            },
        }

        self.assertEqual(loss._compute_dynakey_q_loss(data), {})


if __name__ == "__main__":
    unittest.main()
