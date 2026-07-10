from __future__ import annotations

from gdkvm_project.losses import LossComputer
from losses.base import ce_loss, dice_loss

__all__ = ["LossComputer", "ce_loss", "dice_loss"]
