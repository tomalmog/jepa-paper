"""Small tensor helpers used by the training loop."""
from __future__ import annotations

import torch


@torch.no_grad()
def copy_params(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    for ps, pd in zip(src.parameters(), dst.parameters()):
        pd.data.copy_(ps.data)


@torch.no_grad()
def update_ema(src: torch.nn.Module, dst: torch.nn.Module, momentum: float) -> None:
    """Polyak / EMA update.  ``dst <- m*dst + (1-m)*src``."""
    for ps, pd in zip(src.parameters(), dst.parameters()):
        pd.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum, self.count, self.avg = 0.0, 0, 0.0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)
