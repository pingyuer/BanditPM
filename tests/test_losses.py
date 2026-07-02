from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from losses.base import ce_loss, dice_loss


def test_ce_loss_perfect_prediction():
    logits = torch.zeros(1, 2, 2, 2)
    logits[:, 0] = 10.0
    soft_gt = torch.zeros(1, 2, 2, 2)
    soft_gt[:, 0] = 1.0
    loss = ce_loss(logits, soft_gt)
    assert loss.item() < 0.01


def test_ce_loss_all_wrong():
    logits = torch.zeros(1, 2, 2, 2)
    logits[:, 1] = 10.0
    soft_gt = torch.zeros(1, 2, 2, 2)
    soft_gt[:, 0] = 1.0
    loss = ce_loss(logits, soft_gt)
    assert loss.item() > 1.0


def test_ce_loss_zero_mask_ignored():
    logits = torch.randn(1, 2, 3, 3)
    soft_gt_nonzero = torch.zeros(1, 2, 3, 3)
    soft_gt_nonzero[:, 0] = 1.0
    soft_gt_zero = torch.zeros(1, 2, 3, 3)
    loss_nonzero = ce_loss(logits, soft_gt_nonzero)
    loss_zero = ce_loss(logits, soft_gt_zero)
    assert loss_zero.item() < loss_nonzero.item()
    assert loss_zero.item() == pytest.approx(0.0, abs=1e-6)


def test_dice_loss_perfect_overlap():
    mask = torch.zeros(1, 2, 4, 4)
    mask[:, 1, 0:2, :] = 1.0
    soft_gt = torch.zeros(1, 2, 4, 4)
    soft_gt[:, 1, 0:2, :] = 1.0
    loss = dice_loss(mask, soft_gt)
    assert loss.item() < 0.01


def test_dice_loss_no_overlap():
    mask = torch.zeros(1, 2, 4, 4)
    mask[:, 1, 0:2, :] = 1.0
    soft_gt = torch.zeros(1, 2, 4, 4)
    soft_gt[:, 1, 2:4, :] = 1.0
    loss = dice_loss(mask, soft_gt)
    assert loss.item() > 0.9


def test_dice_loss_partial_overlap():
    mask = torch.zeros(1, 2, 4, 4)
    mask[:, 1, 0:2, :] = 1.0
    soft_gt = torch.zeros(1, 2, 4, 4)
    soft_gt[:, 1, 1:3, :] = 1.0
    loss = dice_loss(mask, soft_gt)
    assert 0.0 < loss.item() < 1.0


def test_dice_loss_empty_mask():
    mask = torch.ones(1, 2, 4, 4) * 0.5
    soft_gt = torch.zeros(1, 2, 4, 4)
    loss = dice_loss(mask, soft_gt)
    assert torch.isfinite(loss)


def test_ce_dice_combined():
    logits = torch.randn(1, 2, 4, 4)
    soft_gt = torch.zeros(1, 2, 4, 4)
    soft_gt[:, 0] = 1.0
    loss_ce = ce_loss(logits, soft_gt)
    probs = F.softmax(logits, dim=1)
    loss_dice = dice_loss(probs, soft_gt)
    combined = loss_ce + loss_dice
    assert torch.isfinite(combined)
    assert combined.item() > 0.0
