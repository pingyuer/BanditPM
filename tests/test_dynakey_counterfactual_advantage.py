import unittest

import torch

from model.modules.dynakey import ODEKeyDictionary, compute_counterfactual_returns


class DynaKeyCounterfactualAdvantageTests(unittest.TestCase):
    def test_advantage_return_matches_keep_error_formula(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=2, ema_alpha=1.0)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        z = torch.tensor([[[0.0, 0.0]]])
        target = torch.tensor([[[1.0, 0.0]]])
        dictionary.spawn(z, torch.zeros_like(z))

        raw, aux = compute_counterfactual_returns(dictionary, z, target)
        self.assertIn("raw_returns", aux)
        self.assertIn("advantage_returns", aux)
        keep_error = aux["keep_error"]
        errors = aux["action_errors"]
        costs = aux["action_cost"]
        expected = keep_error.unsqueeze(-1) - errors - (costs - costs[0])
        self.assertTrue(torch.allclose(aux["advantage_returns"], expected, atol=1e-6))
        self.assertTrue(torch.allclose(raw, aux["raw_returns"]))
        self.assertGreater(aux["advantage_returns"][0, 0, 1].item(), 0.0)
        self.assertLessEqual(aux["advantage_returns"][0, 0, 0].item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
