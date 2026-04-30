import unittest

import torch

from model.modules.dynakey import DynaKeyQMaintainer, ODEKeyDictionary


class DynaKeyQStateTests(unittest.TestCase):
    def test_q_state_shape_and_finite_values(self):
        value_dim = 4
        dictionary = ODEKeyDictionary(value_dim=value_dim, bank_size=3)
        dictionary.reset_state(batch_size=2, num_objects=1, device=torch.device("cpu"))
        maintainer = DynaKeyQMaintainer(value_dim=value_dim, bank_size=3, hidden_dim=16)

        z = torch.randn(2, 1, value_dim)
        dictionary.spawn(z, torch.zeros_like(z))
        weights, retrieval_aux = dictionary.retrieve(z)
        pred, _ = dictionary.predict(z, weights)
        target = z + 0.1

        q_state = maintainer.build_q_state(z, pred, target, dictionary.state, retrieval_aux)

        self.assertEqual(q_state.shape, (2, 1, 2 * value_dim + 16))
        self.assertTrue(torch.isfinite(q_state).all())
        self.assertTrue((q_state[..., maintainer.feature_index["has_target"]] == 1).all())
        self.assertTrue((q_state[..., maintainer.feature_index["l2_error"]] >= 0).all())

    def test_q_state_without_target_is_finite(self):
        value_dim = 4
        dictionary = ODEKeyDictionary(value_dim=value_dim, bank_size=3)
        dictionary.reset_state(batch_size=1, num_objects=2, device=torch.device("cpu"))
        maintainer = DynaKeyQMaintainer(value_dim=value_dim, bank_size=3, hidden_dim=16)

        z = torch.randn(1, 2, value_dim)
        weights, retrieval_aux = dictionary.retrieve(z)
        pred, _ = dictionary.predict(z, weights)
        q_state = maintainer.build_q_state(z, pred, None, dictionary.state, retrieval_aux)

        self.assertEqual(q_state.shape, (1, 2, 2 * value_dim + 16))
        self.assertTrue(torch.isfinite(q_state).all())
        self.assertTrue((q_state[..., maintainer.feature_index["has_target"]] == 0).all())


if __name__ == "__main__":
    unittest.main()
