import unittest

import torch

from model.modules.dynakey.ode_key_dictionary import ODEKeyDictionary, safe_invalid_value


class DynaKeyFp16SafetyTests(unittest.TestCase):
    def test_safe_invalid_value_for_low_precision(self):
        self.assertEqual(safe_invalid_value(torch.float16), -1.0e4)
        self.assertEqual(safe_invalid_value(torch.bfloat16), -1.0e4)
        self.assertEqual(safe_invalid_value(torch.float32), -1.0e9)

    def test_retrieve_with_invalid_slots_is_finite(self):
        dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=3)
        dictionary.reset_state(batch_size=1, num_objects=1, device=device)
        z = torch.tensor([[[1.0, 0.0]]], device=device, dtype=dtype)
        dictionary.spawn(z.float(), torch.zeros_like(z.float()))
        dictionary.state.center = dictionary.state.center.to(dtype)
        dictionary.state.velocity = dictionary.state.velocity.to(dtype)
        dictionary.state.scale = dictionary.state.scale.to(dtype)
        weights, aux = dictionary.retrieve(z)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(weights[0, 0, 1].item(), 0.0)
        self.assertEqual(weights[0, 0, 2].item(), 0.0)
        self.assertEqual(aux["nearest_idx"].item(), 0)


if __name__ == "__main__":
    unittest.main()
