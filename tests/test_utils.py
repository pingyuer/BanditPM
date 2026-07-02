from __future__ import annotations

import unittest
import torch
from utils.tensor_utils import aggregate, cls_to_one_hot, pad_divide_by, unpad


class TestAggregate(unittest.TestCase):

    def test_aggregate_sums_sigmoid_masks(self):
        prob = torch.tensor([[[0.5, 0.5], [0.5, 0.5]],
                             [[0.25, 0.75], [0.1, 0.9]]]).unsqueeze(0)
        result = aggregate(prob, dim=1)
        self.assertEqual(result.shape[1], 3)
        bg_prob = torch.prod(1 - prob, dim=1, keepdim=True)
        combined = torch.cat([bg_prob, prob], dim=1).clamp(1e-7, 1 - 1e-7)
        expected = torch.log(combined / (1 - combined))
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_aggregate_single_class(self):
        prob = torch.tensor([[[0.3, 0.7, 0.5]]]).unsqueeze(0)
        result = aggregate(prob, dim=1)
        self.assertEqual(result.shape, (1, 2, 1, 3))
        bg = (1 - prob).prod(dim=1, keepdim=True)
        combined = torch.cat([bg, prob], dim=1).clamp(1e-7, 1 - 1e-7)
        expected = torch.log(combined / (1 - combined))
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))


class TestClsToOneHot(unittest.TestCase):

    def test_cls_to_one_hot(self):
        cls_gt = torch.tensor([[[0, 1], [2, 0]]]).unsqueeze(0)
        num_objects = 2
        result = cls_to_one_hot(cls_gt, num_objects)
        self.assertEqual(result.shape, (1, 3, 2, 2))
        self.assertEqual(result[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(result[0, 1, 0, 0].item(), 0.0)
        self.assertEqual(result[0, 0, 0, 1].item(), 0.0)
        self.assertEqual(result[0, 1, 0, 1].item(), 1.0)
        self.assertEqual(result[0, 2, 0, 1].item(), 0.0)
        self.assertEqual(result[0, 2, 1, 0].item(), 1.0)
        self.assertEqual(result[0, 0, 1, 1].item(), 1.0)


class TestPadDivideBy(unittest.TestCase):

    def test_pad_divide_by_roundtrip(self):
        t = torch.randn(1, 3, 5, 7)
        padded, pad_info = pad_divide_by(t, 4)
        self.assertEqual(padded.shape[-2] % 4, 0)
        self.assertEqual(padded.shape[-1] % 4, 0)
        restored = unpad(padded, pad_info)
        self.assertEqual(restored.shape, t.shape)
        self.assertTrue(torch.allclose(restored, t))

    def test_pad_divide_by_identity_when_already_divisible(self):
        t = torch.randn(1, 3, 8, 12)
        padded, pad_info = pad_divide_by(t, 4)
        self.assertEqual(padded.shape, t.shape)
        self.assertTrue(torch.allclose(padded, t))
        self.assertEqual(pad_info, (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
