import torch

from rebel.resampling import make_identity_grid, offset_px_to_normalized, sample_feature


def test_rebel_resampling_zero_offset_identity():
    x = torch.randn(2, 3, 6, 7)
    y = sample_feature(x, torch.zeros(2, 2, 6, 7))
    assert torch.allclose(x, y, atol=1e-5)


def test_rebel_resampling_known_positive_x_shift_samples_right_neighbor():
    x = torch.arange(25, dtype=torch.float32).view(1, 1, 5, 5)
    y = sample_feature(x, torch.ones(1, 2, 5, 5) * torch.tensor([1.0, 0.0]).view(1, 2, 1, 1))
    assert torch.allclose(y[0, 0, 2, 2], x[0, 0, 2, 3])


def test_rebel_resampling_known_positive_y_shift_samples_lower_neighbor():
    x = torch.arange(25, dtype=torch.float32).view(1, 1, 5, 5)
    y = sample_feature(x, torch.ones(1, 2, 5, 5) * torch.tensor([0.0, 1.0]).view(1, 2, 1, 1))
    assert torch.allclose(y[0, 0, 2, 2], x[0, 0, 3, 2])


def test_rebel_resampling_align_corners_false_and_border_padding():
    grid = make_identity_grid(1, 3, 5, torch.device("cpu"), torch.float32)
    assert grid.shape == (1, 3, 5, 2)
    off = offset_px_to_normalized(torch.tensor([[[[1.0]], [[1.0]]]]), 3, 5)
    assert torch.allclose(off[:, 0], torch.full((1, 1, 1), 0.4))
    assert torch.allclose(off[:, 1], torch.full((1, 1, 1), 2.0 / 3.0))
    x = torch.randn(1, 1, 3, 5)
    y = sample_feature(x, torch.full((1, 2, 3, 5), 100.0), padding_mode="border")
    assert torch.isfinite(y).all()


def test_rebel_resampling_rectangular_input_uses_separate_hw_scales():
    off = offset_px_to_normalized(torch.tensor([[[[2.0]], [[3.0]]]]), 6, 10)
    assert torch.allclose(off[:, 0], torch.full((1, 1, 1), 0.4))
    assert torch.allclose(off[:, 1], torch.full((1, 1, 1), 1.0))
