import unittest

import torch

from model.modules.dynakey import ODEKeyDictionary


class ODEKeyDictionaryTests(unittest.TestCase):
    def test_init_and_empty_retrieve_predict(self):
        dictionary = ODEKeyDictionary(value_dim=3, bank_size=4)
        dictionary.reset_state(batch_size=2, num_objects=1, device=torch.device("cpu"))

        state = dictionary.state
        self.assertEqual(state.center.shape, (2, 1, 4, 3))
        self.assertFalse(state.valid.any())

        z = torch.randn(2, 1, 3)
        weights, aux = dictionary.retrieve(z)
        pred, pred_aux = dictionary.predict(z, weights)

        self.assertEqual(weights.shape, (2, 1, 4))
        self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))
        self.assertFalse(aux["has_match"].any())
        self.assertTrue(torch.allclose(pred, z))
        self.assertTrue(pred_aux["used_identity_fallback"].all())

    def test_spawn_retrieve_and_euler_predict(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=3, dt=0.5)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))

        z = torch.tensor([[[1.0, 0.0]]])
        velocity = torch.tensor([[[0.0, 2.0]]])
        slot = dictionary.spawn(z, velocity)

        self.assertEqual(slot.shape, (1, 1))
        self.assertTrue(dictionary.state.valid[0, 0, 0])

        weights, aux = dictionary.retrieve(z)
        pred, _ = dictionary.predict(z, weights)

        self.assertTrue(aux["has_match"].all())
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1, 1)))
        self.assertTrue(torch.allclose(pred, torch.tensor([[[1.0, 1.0]]]), atol=1e-5))

    def test_clone_update_split_delete(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=4, ema_alpha=0.5)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))

        z = torch.tensor([[[1.0, 0.0]]])
        target = torch.tensor([[[1.0, 1.0]]])
        dictionary.spawn(z, torch.zeros_like(z))

        dictionary.clone_slot(
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
        )
        self.assertTrue(dictionary.state.valid[0, 0, 1])
        self.assertTrue(torch.allclose(dictionary.state.center[0, 0, 0], dictionary.state.center[0, 0, 1]))

        dictionary.update(z, target, torch.tensor([[0]], dtype=torch.long))
        self.assertGreater(dictionary.state.usage[0, 0, 0].item(), 0.0)
        self.assertGreater(dictionary.state.error_ema[0, 0, 0].item(), 0.0)
        self.assertTrue(torch.allclose(dictionary.state.velocity[0, 0, 0], torch.tensor([0.0, 0.5]), atol=1e-5))

        new_slot = dictionary.split(torch.tensor([[0]], dtype=torch.long), perturb_scale=0.01)
        self.assertEqual(new_slot.item(), 2)
        self.assertTrue(dictionary.state.valid[0, 0, 2])
        self.assertFalse(torch.allclose(dictionary.state.center[0, 0, 0], dictionary.state.center[0, 0, 2]))

        dictionary.delete(torch.tensor([[1]], dtype=torch.long))
        self.assertFalse(dictionary.state.valid[0, 0, 1])
        self.assertEqual(dictionary.state.usage[0, 0, 1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
