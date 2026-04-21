"""
LR / weight-decay / momentum schedulers used by I-JEPA, simplified.

All schedulers are "step every iteration" objects that expose ``.step()``
and have their current value available via ``.val``.
"""
from __future__ import annotations

import math


class _Base:
    val: float

    def step(self) -> float:
        raise NotImplementedError


class WarmupCosineSchedule(_Base):
    """Linear warmup → cosine decay (same as I-JEPA)."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        start_lr: float,
        ref_lr: float,
        final_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self._step = 0
        self.val = start_lr

    def step(self) -> float:
        s = self._step
        if s < self.warmup_steps:
            progress = s / max(1, self.warmup_steps)
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = (s - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)
            lr = self.final_lr + 0.5 * (self.ref_lr - self.final_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.val = lr
        self._step += 1
        return lr


class CosineWDSchedule(_Base):
    """Weight-decay cosine schedule (I-JEPA uses 0.04 → 0.4)."""

    def __init__(self, optimizer, ref_wd: float, final_wd: float, total_steps: int):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.total_steps = total_steps
        self._step = 0
        self.val = ref_wd

    def step(self) -> float:
        progress = min(1.0, self._step / max(1, self.total_steps))
        wd = self.final_wd + 0.5 * (self.ref_wd - self.final_wd) * (
            1.0 + math.cos(math.pi * progress)
        )
        for g in self.optimizer.param_groups:
            if g.get("weight_decay", 0) > 0:
                g["weight_decay"] = wd
        self.val = wd
        self._step += 1
        return wd


class MomentumSchedule(_Base):
    """Linearly increases EMA momentum from m0 to 1.0 over all steps."""

    def __init__(self, ema_start: float, ema_end: float, total_steps: int):
        self.m0 = ema_start
        self.m1 = ema_end
        self.total_steps = total_steps
        self._step = 0
        self.val = ema_start

    def step(self) -> float:
        progress = min(1.0, self._step / max(1, self.total_steps))
        m = self.m0 + progress * (self.m1 - self.m0)
        self.val = m
        self._step += 1
        return m
