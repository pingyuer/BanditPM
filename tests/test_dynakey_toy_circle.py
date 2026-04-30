import math
import unittest

import torch

from model.modules.dynakey import ODEKeyDictionary


class DynaKeyToyCircleTests(unittest.TestCase):
    def test_circle_dynamics_beats_identity_baseline(self):
        torch.manual_seed(7)
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=8, dt=1.0, ema_alpha=0.7)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))

        angles = torch.linspace(0, math.pi, steps=12)
        states = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).view(12, 1, 1, 2)

        pred_errors = []
        identity_errors = []
        for t in range(states.shape[0] - 1):
            z = states[t]
            target = states[t + 1]
            weights, _ = dictionary.retrieve(z)
            pred, pred_aux = dictionary.predict(z, weights)
            pred_errors.append(torch.mean((pred - target) ** 2))
            identity_errors.append(torch.mean((z - target) ** 2))

            if pred_aux["used_identity_fallback"].any() or dictionary.state.valid.sum() < 4:
                dictionary.spawn(z, target - z)
            else:
                nearest = dictionary.retrieve(z)[1]["nearest_idx"]
                dictionary.update(z, target, nearest)

        self.assertLess(torch.stack(pred_errors[4:]).mean().item(), torch.stack(identity_errors[4:]).mean().item())


if __name__ == "__main__":
    unittest.main()
