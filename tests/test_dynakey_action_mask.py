import unittest

import torch

from model.modules.dynakey import DynaKeyQMaintainer, ODEKeyDictionary


class DynaKeyActionMaskTests(unittest.TestCase):
    def test_empty_dictionary_allows_keep_and_spawn_only(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=2)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        maintainer = DynaKeyQMaintainer(value_dim=2, bank_size=2, hidden_dim=8)
        z = torch.zeros(1, 1, 2)
        _, aux = dictionary.retrieve(z)

        mask = maintainer.action_mask(dictionary.state, aux)

        self.assertTrue(torch.equal(mask[0, 0], torch.tensor([True, False, True, False, False])))

    def test_full_dictionary_disallows_spawn_and_split(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=2)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        maintainer = DynaKeyQMaintainer(value_dim=2, bank_size=2, hidden_dim=8)
        z = torch.zeros(1, 1, 2)
        dictionary.spawn(z, torch.zeros_like(z))
        dictionary.spawn(z + 1, torch.zeros_like(z))
        _, aux = dictionary.retrieve(z)

        mask = maintainer.action_mask(dictionary.state, aux)

        self.assertTrue(torch.equal(mask[0, 0], torch.tensor([True, True, False, False, True])))

    def test_greedy_selection_never_selects_invalid_action(self):
        maintainer = DynaKeyQMaintainer(value_dim=2, bank_size=2, hidden_dim=8)
        q_values = torch.tensor([[[0.0, 100.0, 2.0, 99.0, 98.0]]])
        mask = torch.tensor([[[True, False, True, False, False]]])

        action = maintainer.select_action(q_values, mask, mode="greedy")

        self.assertEqual(action.item(), 2)


if __name__ == "__main__":
    unittest.main()
