"""
SIGReg (Sketched Isotropic Gaussian Regularization) for Hyperbolic-JEPA.

Implements the anti-collapse objective from Balestriero & LeCun (2025)
"LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
(arXiv:2511.08544). Ported from the reference implementation at
github.com/galilai-group/lejepa (CC-BY-SA 4.0).

The loss projects batch features onto M random unit directions and applies
the Epps-Pulley normality test to each 1D projection. By Cramer-Wold,
matching all 1D marginals implies matching the joint distribution, so
SIGReg → 0 <=> Z → isotropic Gaussian.

For hyperbolic models, tangent_sigreg applies SIGReg to log₀(z), targeting
an isotropic Gaussian in the tangent space at the origin. The Poincaré
ball is bounded so ambient-Gaussian matching is ill-posed; the tangent
space is unbounded Euclidean and admits a natural Gaussian prior.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.hyperbolic import clip_feature, logmap0


class EppsPulley(nn.Module):
    """Epps-Pulley normality test statistic (differentiable).

    Compares the empirical characteristic function of the input samples
    against the standard normal CF using a weighted L2 distance:

        T = N * integral |phi_hat(t) - exp(-t²/2)|² * exp(-t²/2) dt

    The integrand is symmetric in t, so we integrate over [0, t_max] and
    absorb the factor of 2 into the quadrature weights (endpoint halved).

    Input:  (*, N, K) — N samples of K univariate projections
    Output: (*, K) — one statistic per projection
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1, "n_points should be odd"
        t = torch.linspace(0.0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        # Trapezoid weights, doubled (symmetry), halved at endpoints
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        phi = (-0.5 * t.square()).exp()  # standard normal CF on real line
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)  # pre-multiply weight function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(-2)
        # Shape: (*, N, K, n_points)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = torch.cos(x_t).mean(dim=-3)
        sin_mean = torch.sin(x_t).mean(dim=-3)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        # err: (*, K, n_points), weights: (n_points,)
        return (err @ self.weights) * N


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization.

    Projects batch features Z ∈ R^{N×D} onto M random unit directions
    and averages the Epps-Pulley statistic across projections.

    Directions are resampled every forward pass (seeded by an internal
    counter for reproducibility). Paper shows resampling per-step with
    M=16–256 outperforms a fixed set of thousands.

    Parameters
    ----------
    num_slices : int
        M, the number of random directions. LeJEPA default 256; LeWM uses 1024.
    t_max, n_points : passed to EppsPulley.
    """

    def __init__(
        self,
        num_slices: int = 256,
        t_max: float = 3.0,
        n_points: int = 17,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.ep = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (N, D) or (*, N, D)

        Returns
        -------
        Scalar loss = mean over slices of per-slice Epps-Pulley statistic.
        """
        with torch.no_grad():
            seed = int(self.global_step.item())
            g = self._get_generator(z.device, seed)
            A = torch.randn(z.size(-1), self.num_slices, device=z.device, generator=g)
            A = A / A.norm(p=2, dim=0, keepdim=True)  # unit-norm columns
            self.global_step.add_(1)
        # z @ A: (*, N, num_slices) -- univariate projections
        stats = self.ep(z @ A)
        return stats.mean()


class TangentSIGReg(nn.Module):
    """SIGReg applied in the tangent space at the origin of the Poincaré ball.

    z ∈ ball -> log₀(z) ∈ R^D -> SIGReg -> scalar.

    For Euclidean features (c irrelevant), pass curvature=None to skip logmap.
    """

    def __init__(
        self,
        curvature: float | None,
        clip_radius: float = 1.0,
        num_slices: int = 256,
        t_max: float = 3.0,
        n_points: int = 17,
    ):
        super().__init__()
        self.curvature = curvature
        self.clip_radius = clip_radius
        self.sigreg = SIGReg(num_slices=num_slices, t_max=t_max, n_points=n_points)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.curvature is not None:
            z = clip_feature(z, self.clip_radius)
            z = logmap0(z, self.curvature)
        return self.sigreg(z)


def build_sigreg(cfg: dict) -> nn.Module | None:
    """
    Build a SIGReg module from config.

    Config keys:
        sigreg: bool                    - enable/disable
        sigreg_mode: "ambient" | "tangent" (default "tangent")
        sigreg_num_slices: int          - M (default 256)
        sigreg_t_max: float             - EP integration range (default 3.0)
        sigreg_n_points: int            - EP quadrature knots (default 17)
        curvature: float                - ball curvature c (needed for tangent)
        clip_radius: float              - norm clip before logmap (default 1.0)
    """
    if not cfg.get("sigreg", False):
        return None
    mode = cfg.get("sigreg_mode", "tangent")
    num_slices = int(cfg.get("sigreg_num_slices", 256))
    t_max = float(cfg.get("sigreg_t_max", 3.0))
    n_points = int(cfg.get("sigreg_n_points", 17))
    if mode == "ambient":
        return SIGReg(num_slices=num_slices, t_max=t_max, n_points=n_points)
    if mode == "tangent":
        curv = float(cfg.get("curvature", 1.0))
        clip = float(cfg.get("clip_radius", 1.0))
        return TangentSIGReg(
            curvature=curv,
            clip_radius=clip,
            num_slices=num_slices,
            t_max=t_max,
            n_points=n_points,
        )
    raise ValueError(f"Unknown sigreg_mode: {mode}")
