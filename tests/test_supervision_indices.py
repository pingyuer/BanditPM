import unittest

import torch

from model.trainer import Trainer


class SupervisionIndexTests(unittest.TestCase):
    def test_dense_label_valid_uses_all_frames(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        data = {
            "rgb": torch.zeros(2, 4, 1, 8, 8),
            "label_valid": torch.ones(2, 4, dtype=torch.bool),
        }
        idx = Trainer._resolve_supervised_indices(trainer, data)
        self.assertTrue(torch.equal(idx, torch.tensor([0, 1, 2, 3])))

    def test_sparse_label_valid_keeps_selected_frames(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        data = {
            "rgb": torch.zeros(1, 5, 1, 8, 8),
            "label_valid": torch.tensor([[True, False, False, False, True]]),
        }
        idx = Trainer._resolve_supervised_indices(trainer, data)
        self.assertTrue(torch.equal(idx, torch.tensor([0, 4])))


if __name__ == "__main__":
    unittest.main()
