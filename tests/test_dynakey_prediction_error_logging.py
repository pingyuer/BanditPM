import unittest

import torch

from model.modules.dynakey import DynaKeyMemoryCore


def value_from_z(z):
    return z.view(1, 1, 2, 1, 1).expand(1, 1, 2, 2, 2).contiguous()


class DynaKeyPredictionErrorLoggingTests(unittest.TestCase):
    def test_prediction_error_uses_previous_prediction(self):
        core = DynaKeyMemoryCore({"BANK_SIZE": 3, "POLICY_MODE": "no_update", "GATE_INIT": 1.0}, value_dim=2)
        core.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        mask = torch.ones(1, 1, 2, 2)

        _, aux0 = core(value_from_z(torch.tensor([0.0, 0.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        self.assertEqual(aux0["prediction_error"].item(), 0.0)

        core.dictionary.state.velocity[0, 0, 0] = torch.tensor([10.0, 0.0])
        _, aux1 = core(value_from_z(torch.tensor([0.0, 1.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        self.assertGreater(aux1["prediction_error"].item(), 0.0)
        self.assertGreater(aux1["residual_norm"].item(), 0.0)

        _, aux2 = core(value_from_z(torch.tensor([0.0, 2.0])), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
        self.assertGreater(aux2["prediction_error"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
