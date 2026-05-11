"""
Tangent-space VICReg regularization for Hyperbolic-JEPA.

Implements the variance-invariance-covariance regularization (Bardes et al. 2022)
applied in the tangent space at the origin of the Poincaré ball.

For Euclidean models (Model A), this is standard VICReg on the raw features.
For hyperbolic models (Model B/C), we apply log₀ first to get tangent coordinates,
then run VICReg in that Euclidean tangent space.

This keeps the regularization consistent across all three model variants —
all see VICReg in the same coordinate system.

Reference:
    Bardes, Ponce, LeCun. "VICReg: Variance-Invariance-Covariance Regularization
    for Self-Supervised Learning." ICLR 2022.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Variance term: force per-dimension std >= gamma.

    Hinge loss on std: max(0, gamma - std(z_d)).
    """
    std = z.std(dim=0)  # (D,)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Covariance term: decorrelate dimensions.

    Off-diagonal elements of the covariance matrix should be zero.
    """
    N, D = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (N - 1)  # (D, D)
    # Zero out diagonal, sum squares of off-diagonal
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D


class VICRegLoss(torch.nn.Module):
    """VICReg applied to encoder representations.

    Applied to the *context encoder* output (mean-pooled patch tokens)
    to prevent dimensional collapse during JEPA training.

    Parameters
    ----------
    var_weight : float
        Weight for variance term (default 25.0 from VICReg paper).
    cov_weight : float
        Weight for covariance term (default 1.0 from VICReg paper).
    gamma : float
        Target std for variance hinge (default 1.0).
    """

    def __init__(
        self,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, D) — mean-pooled encoder features (already in tangent/Euclidean space)

        Returns
        -------
        Scalar loss = var_weight * variance_loss + cov_weight * covariance_loss
        """
        var_loss = variance_loss(z, self.gamma)
        cov_loss = covariance_loss(z)
        return self.var_weight * var_loss + self.cov_weight * cov_loss


def build_vicreg(cfg: dict) -> VICRegLoss | None:
    """Build VICReg loss from config. Returns None if vicreg is disabled."""
    if not cfg.get("vicreg", False):
        return None
    return VICRegLoss(
        var_weight=cfg.get("vicreg_var_weight", 25.0),
        cov_weight=cfg.get("vicreg_cov_weight", 1.0),
        gamma=cfg.get("vicreg_gamma", 1.0),
    )
