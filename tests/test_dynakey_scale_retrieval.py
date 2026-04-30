import unittest

import torch

from model.modules.dynakey import ODEKeyDictionary


class DynaKeyScaleRetrievalTests(unittest.TestCase):
    def test_scale_changes_retrieval_weights(self):
        dictionary = ODEKeyDictionary(value_dim=2, bank_size=2, retrieval_temperature=1.0)
        dictionary.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        dictionary.spawn(torch.tensor([[[1.0, 0.0]]]), torch.zeros(1, 1, 2), torch.tensor([[0]]))
        dictionary.spawn(torch.tensor([[[-1.0, 0.0]]]), torch.zeros(1, 1, 2), torch.tensor([[1]]))
        query = torch.tensor([[[0.0, 1.0]]])

        dictionary.state.scale[0, 0, 0] = 0.5
        dictionary.state.scale[0, 0, 1] = 2.0
        weights_large_slot1, _ = dictionary.retrieve(query)

        dictionary.state.scale[0, 0, 0] = 2.0
        dictionary.state.scale[0, 0, 1] = 0.5
        weights_large_slot0, _ = dictionary.retrieve(query)

        self.assertGreater(weights_large_slot1[0, 0, 1].item(), weights_large_slot1[0, 0, 0].item())
        self.assertGreater(weights_large_slot0[0, 0, 0].item(), weights_large_slot0[0, 0, 1].item())


if __name__ == "__main__":
    unittest.main()
