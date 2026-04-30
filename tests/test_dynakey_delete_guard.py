import unittest

import torch

from model.modules.dynakey import DynaKeyQMaintainer, ODEKeyDictionary


class DynaKeyDeleteGuardTests(unittest.TestCase):
    def test_delete_mask_and_direct_delete_preserve_last_key(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=3)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        z = torch.tensor([[[1.0, 0.0]]])
        dictionary.spawn(z, torch.zeros_like(z))
        weights, aux = dictionary.retrieve(z)
        maintainer = DynaKeyQMaintainer(value_dim=2, bank_size=3, hidden_dim=8)
        mask = maintainer.action_mask(dictionary.state, {**aux, "weights": weights})

        self.assertFalse(mask[0, 0, maintainer.ACTION_DELETE].item())
        dictionary.delete(torch.tensor([[0]]))
        self.assertGreaterEqual(dictionary.active_key_count().item(), 1)
        self.assertTrue(dictionary.state.valid[0, 0, 0])


if __name__ == "__main__":
    unittest.main()
