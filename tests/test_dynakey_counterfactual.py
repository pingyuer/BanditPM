import unittest

import torch

from model.modules.dynakey import ODEKeyDictionary, compute_counterfactual_returns


class DynaKeyCounterfactualTests(unittest.TestCase):
    def test_returns_are_finite_and_do_not_mutate_live_state(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=2)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        z = torch.tensor([[[0.0, 0.0]]])
        target = torch.tensor([[[1.0, 0.0]]])
        dictionary.spawn(z, torch.zeros_like(z))
        before_center = dictionary.state.center.clone()
        before_valid = dictionary.state.valid.clone()

        returns, aux = compute_counterfactual_returns(dictionary, z, target)

        self.assertEqual(returns.shape, (1, 1, 5))
        self.assertTrue(torch.isfinite(returns).all())
        self.assertEqual(aux["prediction_error"].shape, (1, 1, 5))
        self.assertTrue(torch.allclose(dictionary.state.center, before_center))
        self.assertTrue(torch.equal(dictionary.state.valid, before_valid))

    def test_update_or_spawn_improves_known_transition_over_keep(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=2)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        z = torch.tensor([[[0.0, 0.0]]])
        target = torch.tensor([[[1.0, 0.0]]])
        dictionary.spawn(z, torch.zeros_like(z))

        returns, _ = compute_counterfactual_returns(dictionary, z, target)
        keep_return = returns[0, 0, 0]
        best_mutating = torch.max(returns[0, 0, 1:3])

        self.assertGreater(best_mutating.item(), keep_return.item())

    def test_action_costs_lower_equal_predictions(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=2)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        z = torch.tensor([[[0.0, 0.0]]])
        target = z.clone()

        returns, _ = compute_counterfactual_returns(dictionary, z, target)

        self.assertGreater(returns[0, 0, 0].item(), returns[0, 0, 2].item())


if __name__ == "__main__":
    unittest.main()
