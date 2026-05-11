---
name: SIGReg reference — paper and code
description: Sources and exact spec for Sketched Isotropic Gaussian Regularization, used for B+SIGReg experiments
type: reference
originSessionId: 7f294af5-ddf3-4265-925e-bd7bd9f8a494
---
## Primary sources

- **Original paper**: Balestriero & LeCun, *"LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"*, arXiv:2511.08544 (2025). Section 4 defines SIGReg; Algorithm 1 gives full PyTorch code; Algorithm 2 gives LeJEPA (SIGReg + prediction loss).
- **Application paper (image-SSL use)**: Maes, Le Lidec, Scieur, LeCun, Balestriero — *LeWorldModel*, arXiv:2603.19312 (2026). Uses SIGReg with M=1024, λ=0.1, t-range [0.2, 4].
- **Reference code**: https://github.com/galilai-group/lejepa (default branch `main`, 1k+ stars).
  - Multivariate wrapper: `lejepa/multivariate/slicing.py` → `SlicingUnivariateTest`
  - Epps-Pulley: `lejepa/univariate/epps_pulley.py` → `EppsPulley` class (fast version, not `DeprecatedEppsPulley`)

## Exact SIGReg spec

- **Loss**: `SIGReg(Z) = (1/M) * Σ T(Z @ u_m)` where `u_m ~ Uniform(S^{d-1})` and T = Epps-Pulley statistic.
- **Sampling**: resample M directions **every forward pass** (seeded by a synced global step in DDP). Paper shows resampling >> fixed for the same M budget.
- **Directions**: `A = randn(d, M); A /= A.norm(dim=0)` — Gaussian then normalize → uniform on hypersphere.
- **Epps-Pulley (fast version)**:
  - Integrate `|φ̂(t) - φ_N(t)|² · exp(-t²/2)` over `t ∈ [0, t_max]` (uses symmetry to halve cost; weights doubled internally).
  - `φ̂(t) = mean_n(exp(i·t·(Z·u_m)_n))` → compute `cos_mean`, `sin_mean` separately.
  - `err = (cos_mean - exp(-t²/2))² + sin_mean²`, then trapezoidal weighted integration × N.
- **Defaults**: `t_max = 3`, `n_points = 17` (odd required for Simpson; trapezoid used in production), `num_slices (M) = 256` in reference code; paper uses M=1024 in LeWM experiments.
- **Weighting function**: Gaussian, `w(t) = exp(-t²/σ²)` with `σ = 1` → absorbed into the `phi = exp(-t²/2)` precompute.
- **Placement**: Apply to **projector output**, not raw encoder. LeWM: the projector is a 1-layer MLP with BatchNorm after the encoder's [CLS] pool. This is because LayerNorm on the final ViT layer would prevent anti-collapse from biting (mean/var are re-normalized away). **Critical implementation note for our code.**
- **Loss weight**: `L = (1-λ)·L_pred + λ·SIGReg` in LeJEPA formulation; LeWM uses `L = L_pred + λ·SIGReg` with λ=0.1. Robust across λ ∈ [0.01, 0.2]; degrades at λ ≥ 0.5.

## Our adaptation (tangent-SIGReg for Poincaré)

- Apply SIGReg to `log_0(z)` instead of `z` directly. Rationale: the Poincaré ball is bounded → Gaussian target is ill-posed in ambient coords. Tangent at origin is unbounded Euclidean → Gaussian is a natural target.
- Replace the projector's final layer with an identity → logmap_0 so the anti-collapse signal applies in the same space used by the Poincaré distance.
- Keep everything else (M, n_points, t_max, λ) at LeJEPA defaults for the first smoke test.

**Why this matters**: we cite LeJEPA as a foundational prior and position tangent-SIGReg as the hyperbolic-specific methodological contribution.

**How to apply**: When implementing `src/sigreg.py`, mirror the reference impl's `EppsPulley` + `SlicingUnivariateTest` composition. Don't reinvent — port.
