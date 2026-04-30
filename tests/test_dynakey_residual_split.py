import unittest

import torch
import torch.nn.functional as F

from model.modules.dynakey import ODEKeyDictionary


class DynaKeyResidualSplitTests(unittest.TestCase):
    def test_split_uses_residual_direction(self):
        dictionary = ODEKeyDictionary(value_dim=3, bank_size=3, dt=1.0)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        z = torch.tensor([[[0.0, 0.0, 0.0]]])
        dictionary.spawn(z, torch.zeros_like(z))
        residual = torch.tensor([[[0.0, 2.0, 0.0]]])
        new_slot = dictionary.split(torch.tensor([[0]]), residual=residual, split_eps=0.2)

        delta = dictionary.state.center[0, 0, new_slot.item()] - dictionary.state.center[0, 0, 0]
        direction = F.normalize(residual, dim=-1)[0, 0]
        self.assertGreater(torch.dot(F.normalize(delta, dim=0), direction).item(), 0.99)
        self.assertGreater(abs(delta[1].item()), abs(delta[0].item()))

    def test_split_random_fallback_not_fixed_first_dimension(self):
        torch.manual_seed(123)
        dictionary = ODEKeyDictionary(value_dim=4, bank_size=3)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        dictionary.spawn(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
        new_slot = dictionary.split(torch.tensor([[0]]), residual=None, split_eps=0.2)
        delta = dictionary.state.center[0, 0, new_slot.item()] - dictionary.state.center[0, 0, 0]
        self.assertGreater(delta[1:].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
