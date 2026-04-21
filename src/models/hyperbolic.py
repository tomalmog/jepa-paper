"""
Poincaré-ball hyperbolic operations for Hyperbolic-JEPA.

References
----------
- Ganea, Becigneul, Hofmann. "Hyperbolic Neural Networks." NeurIPS 2018.
- Khrulkov, Mirvakhabova, Ustinova, Oseledets, Lempitsky.
  "Hyperbolic Image Embeddings." CVPR 2020.
- Guo, Wang, Tang, Yeung. "Free Hyperbolic Neural Networks with Limited
  Radius." ICML 2022 (a.k.a. "feature clipping" trick).
- Nickel & Kiela. "Poincaré Embeddings for Learning Hierarchical
  Representations." NeurIPS 2017.

Design
------
Only the *loss* and a final projection layer live in hyperbolic space.
All parameters stay Euclidean, so we use standard AdamW — no Riemannian
optimizer needed.  This matches the setup of Khrulkov et al. (CVPR 2020)
and keeps the diff against I-JEPA minimal.
"""
from __future__ import annotations

import torch

# --- numerical constants -----------------------------------------------------
MIN_NORM = 1e-15           # avoid div-by-zero on norms
BALL_EPS = 1e-5            # keep points strictly inside the ball
MAX_TANH_ARG = 15.0        # prevent tanh saturation / NaN


# -----------------------------------------------------------------------------
# Projection & feature clipping
# -----------------------------------------------------------------------------
def project(x: torch.Tensor, c: float) -> torch.Tensor:
    """Project a point onto the open Poincaré ball of curvature -c.

    Ensures ``sqrt(c) * ||x|| < 1 - BALL_EPS``.
    """
    norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    max_norm = (1.0 - BALL_EPS) / (c ** 0.5)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def clip_feature(x: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
    """Euclidean feature clipping from Guo et al. 2022.

    Bounds ``||x||_2 <= max_norm`` *before* the exp-map, which empirically
    eliminates gradient blow-up near the ball boundary.
    """
    norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


# -----------------------------------------------------------------------------
# Exponential / logarithmic maps at the origin
# -----------------------------------------------------------------------------
def expmap0(u: torch.Tensor, c: float) -> torch.Tensor:
    r"""Exp map at origin on the Poincaré ball.

    .. math:: \exp_0^c(u) = \tanh(\sqrt{c}\,\|u\|) \, \frac{u}{\sqrt{c}\,\|u\|}
    """
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    scaled = (sqrt_c * u_norm).clamp(max=MAX_TANH_ARG)
    gamma = torch.tanh(scaled) * u / (sqrt_c * u_norm)
    return project(gamma, c)


def logmap0(x: torch.Tensor, c: float) -> torch.Tensor:
    r"""Log map at origin on the Poincaré ball.

    .. math:: \log_0^c(x) = \mathrm{arctanh}(\sqrt{c}\,\|x\|)
             \, \frac{x}{\sqrt{c}\,\|x\|}
    """
    sqrt_c = c ** 0.5
    x = project(x, c)
    x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    arg = (sqrt_c * x_norm).clamp(max=1.0 - BALL_EPS)
    return x / (sqrt_c * x_norm) * torch.atanh(arg)


# -----------------------------------------------------------------------------
# Möbius addition (ball group operation)
# -----------------------------------------------------------------------------
def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    r"""Möbius addition on the Poincaré ball.

    .. math::
       x \oplus_c y =
       \frac{(1 + 2c\langle x,y\rangle + c\|y\|^2) x + (1 - c\|x\|^2) y}
            {1 + 2c\langle x,y\rangle + c^2\|x\|^2\|y\|^2}
    """
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denom = 1.0 + 2.0 * c * xy + (c ** 2) * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


# -----------------------------------------------------------------------------
# Distance
# -----------------------------------------------------------------------------
def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    r"""Geodesic distance on the Poincaré ball.

    .. math::
       d_c(x, y) = \frac{2}{\sqrt{c}}
       \; \mathrm{arctanh}\!\big(\sqrt{c}\,\|(-x)\oplus_c y\|\big)

    Shape: ``x, y -> (...,)``  (last dim of x / y is the ambient dim).
    """
    sqrt_c = c ** 0.5
    mob = mobius_add(-x, y, c)
    norm = mob.norm(dim=-1, p=2).clamp(max=(1.0 - BALL_EPS) / sqrt_c)
    return (2.0 / sqrt_c) * torch.atanh((sqrt_c * norm).clamp(max=1.0 - BALL_EPS))


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
class PoincareRegressionLoss(torch.nn.Module):
    """Drop-in replacement for I-JEPA's smooth-L1 loss in hyperbolic space.

    ``pred`` and ``target`` are Euclidean feature vectors (ViT outputs).
    We (1) clip them to a Euclidean ball of radius ``clip_radius``,
    (2) map them onto the Poincaré ball via exp_0, and (3) penalise
    their squared geodesic distance.

    Squaring mirrors ``F.mse_loss``; if you want a smoother objective
    set ``power=1.0`` for pure distance (closer to smooth-L1).
    """

    def __init__(
        self,
        curvature: float = 1.0,
        clip_radius: float = 1.0,
        power: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if curvature <= 0:
            raise ValueError("curvature must be > 0 (we use c = -K for K<0).")
        self.c = float(curvature)
        self.clip_radius = float(clip_radius)
        self.power = float(power)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # (1) feature clipping for stability
        pred = clip_feature(pred, self.clip_radius)
        target = clip_feature(target, self.clip_radius)

        # (2) map onto Poincaré ball
        pred_h = expmap0(pred, self.c)
        target_h = expmap0(target, self.c)

        # (3) Poincaré-distance-based loss
        dist = poincare_distance(pred_h, target_h, self.c)
        loss = dist.pow(self.power)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class EuclideanIJEPALoss(torch.nn.Module):
    """Faithful replica of I-JEPA's Euclidean regression loss.

    This matches facebookresearch/ijepa exactly::

        loss = mean(|pred - target| ** power) / power

    The default ``power=1.0`` gives the L1/smooth-L1-style loss used in
    the paper. ``power=2.0`` gives the half-MSE variant, which is what
    you want for an A/B against a squared Poincaré distance.
    """

    def __init__(self, power: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.power = float(power)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target).abs().pow(self.power) / self.power
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg: dict) -> torch.nn.Module:
    """Factory: pick the loss from cfg['loss_type'] ('poincare' | 'euclidean').

    Defaults to 'poincare' so the pre-existing hyperbolic configs keep
    working without modification.
    """
    loss_type = str(cfg.get("loss_type", "poincare")).lower()
    if loss_type == "poincare":
        return PoincareRegressionLoss(
            curvature=cfg["curvature"],
            clip_radius=cfg["clip_radius"],
            power=cfg["loss_power"],
        )
    if loss_type == "euclidean":
        return EuclideanIJEPALoss(power=cfg["loss_power"])
    raise ValueError(
        f"Unknown loss_type {loss_type!r}. Expected 'poincare' or 'euclidean'."
    )
