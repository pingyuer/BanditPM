from losses.base import ce_loss, dice_loss


def __getattr__(name: str):
    if name == "LossComputer":
        from gdkvm_project.losses import LossComputer

        return LossComputer
    raise AttributeError(name)

__all__ = ["LossComputer", "ce_loss", "dice_loss"]
