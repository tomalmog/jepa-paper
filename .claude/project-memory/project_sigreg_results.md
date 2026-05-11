---
name: SIGReg experimental matrix results (COMPLETE)
description: Final 36-job results from the {A,B} × {VICReg,SIGReg} × {C10,C100,TinyIN} × BN projector matrix completed 2026-04-22
type: project
originSessionId: 7f294af5-ddf3-4265-925e-bd7bd9f8a494
---
## Matrix design

All 36 runs use a BN projector head before the anti-collapse regularizer (per LeJEPA Sec 5.1 — LayerNorm at ViT output kills anti-collapse signal).

- **A** = Euclidean encoder + Euclidean prediction loss
- **B** = Euclidean encoder + Poincaré prediction loss (curvature=1.0, clip=1.0)
- **VICReg** = variance/invariance/covariance regularizer, weight 0.1
- **SIGReg** = LeJEPA's Sketched Isotropic Gaussian Regularizer; B=tangent space, A=ambient. M=256, t_max=3, n_points=17, weight 0.1
- All ViT-Tiny (192-dim), 3 seeds (42/123/7)

## Full results — linear probe accuracy (Euclidean pool)

### CIFAR-10, 50 epochs

| Config | s42 | s123 | s7 | Mean | Collapse (rank / mv) |
|---|---|---|---|---|---|
| A+VICReg+proj | 42.84 | 41.89 | 20.90 | **35.2%** (s7 outlier) | ~161 / 0.49 |
| A+SIGReg+proj | 30.97 | 31.62 | 25.88 | **29.5%** | ~70 / 0.11 |
| B+VICReg+proj | 44.86 | 44.79 | 41.40 | **43.7%** ⭐ | ~136 / 0.53 |
| B+SIGReg+proj | 33.73 | 33.73 | 35.87 | **34.4%** | ~32 / 0.35 |

### CIFAR-100, 200 epochs

| Config | s42 | s123 | s7 | Mean | Collapse (rank / mv) |
|---|---|---|---|---|---|
| A+VICReg+proj | 17.19 | 19.11 | 17.60 | **18.0%** | ~181 / 0.60 |
| A+SIGReg+proj | 10.72 | 9.06 | 5.94 | **8.6%** (degraded) | ~79 / 0.42 |
| B+VICReg+proj | 21.39 | 19.86 | 17.51 | **19.6%** ⭐ | ~153 / 0.36 |
| B+SIGReg+proj | 18.55 | 19.77 | 19.83 | **19.4%** | ~78 / 0.53 |

### Tiny-ImageNet, 300 epochs, img_size 64, patch 8

| Config | s42 | s123 | s7 | Mean | Collapse (rank / mv) |
|---|---|---|---|---|---|
| A+VICReg+proj | 8.64 | 6.58 | 8.28 | **7.8%** | ~166 / 0.36 |
| A+SIGReg+proj | 3.95 | 0.50 | 0.95 | **1.8% (collapsed)** | ~18 / 0.03 (nan for s123) |
| B+VICReg+proj | 10.84 | 11.25 | 11.35 | **11.1%** | ~142 / 0.09 |
| B+SIGReg+proj | 13.40 | 14.79 | 13.75 | **14.0%** ⭐ | ~88 / 0.60 |

## Key findings — scale-dependent regularizer choice

**Clean monotonic trend**: B+VICReg vs B+SIGReg gap closes with scale, and flips on TinyIN.

| Dataset (scale) | B+VICReg | B+SIGReg | Winner, margin |
|---|---|---|---|
| CIFAR-10 (50ep, 50K imgs) | 43.7% | 34.4% | VICReg +9.3 pp |
| CIFAR-100 (200ep, 50K imgs) | 19.6% | 19.4% | Tied |
| Tiny-ImageNet (300ep, 100K imgs) | 11.1% | **14.0%** | SIGReg +2.9 pp |

## Core conclusions

1. **Poincaré predictor (B) beats Euclidean (A) across every dataset × regularizer combination.** Zero exceptions. This is the paper's main claim, now on rock-solid ground.
2. **Tangent-SIGReg (B+SIGReg) becomes the winner at Tiny-ImageNet scale.** Not a fluke — 3/3 seeds beat all B+VICReg seeds. Lowest B+SIGReg (13.40) > highest B+VICReg (11.35).
3. **Ambient SIGReg (A+SIGReg) is catastrophic at TinyIN scale.** A+SIGReg seed 123 fully diverged (NaN metrics), rank 1.6 on seed 7 (complete collapse). The Poincaré tangent structure is what stabilizes SIGReg — neither component alone works at scale.
4. **Rank and mean_var metrics** are informative but misleading on their own. B+SIGReg has LOWER rank (~88) than B+VICReg (~142) on TinyIN, yet higher probe accuracy — anti-collapse by rank alone is not the right objective.
5. **BN projector costs 3-4pp** on C10/C100 vs no-projector baselines (from prior runs). On TinyIN we don't have no-proj comparison for B+VICReg or B+SIGReg, but presumably similar.

## Paper positioning (revised)

**Original claim**: "Hyperbolic JEPA + VICReg wins at small ViT scale"
**Updated claim**: "Poincaré predictor wins universally. The right regularizer depends on scale — VICReg at small scale, SIGReg (applied to tangent space) at larger scale. Their composition with the hyperbolic head is what enables both."

**This is a stronger paper**:
- Two novel contributions that compose (B + tangent-SIGReg)
- Clean positive + negative results for every cell
- A concrete, scale-dependent prescription for which regularizer to use
- SIGReg/LeJEPA position: we strengthen LeJEPA's claim (needs scale) while showing its ambient form destabilizes on small-scale ViTs; only the tangent formulation works.

## Follow-up matrix (in progress, 2026-04-23)

24-job matrix to strengthen TinyIN scaling + de-confound projector. Early partial results:

**TinyIN 600ep (extending scaling curve)**:
| Config | s42 | s123 | s7 | Mean (300ep baseline) |
|---|---|---|---|---|
| B+VICReg+proj 600ep | 10.85 | 12.63 | 12.32 | **11.93%** (300ep=11.1%, +0.8pp — flat) |
| B+SIGReg+proj 600ep | 17.60 | 17.01 | 15.71 | **16.77%** (300ep=14.0%, +2.8pp) |

**B-cell 600ep complete (6/6 seeds). Gap widens from +2.9pp (300ep) → +4.84pp (600ep).** B+SIG wins 3/3 seeds; lowest B+SIG (15.71) > highest B+VIC (12.63). Clean scale-dependent trend — SIGReg benefits from extra compute, VICReg plateaus. This is the headline finding for the paper.

**No-projector ablation COMPLETE (6/6 seeds)**: the projector effect is opposite for the two methods, and flips the winner:

| Config | with proj (300ep mean) | no proj (300ep) |
|---|---|---|
| B+VICReg | 11.1% | **14.70%** (s42=15.20, s123=14.24, s7=14.67) — proj *hurts* VIC by 3.6pp |
| B+SIGReg | 14.0% | **11.33%** (11.25/11.30/11.45) — proj *helps* SIG by 2.7pp |

**Without projector, B+VIC beats B+SIG by +3.4pp at 300ep** — every B+VIC seed beats every B+SIG seed. Clean separation. The 300ep "SIGReg wins" result was projector-dependent. **With projector** at 600ep, B+SIG still wins by +4.84pp. Paper story: projector × regularizer interaction is real; our best config is B+SIG+projector at scale. Cannot claim "SIGReg > VICReg universally" — it's conditional on projector. 600ep+noproj matrix would be needed to fully resolve, but compute-expensive.

**A+SIG hparam debug — COMPLETE**:

| Variant | s42 | s123 | s7 | Mean | Rank / mv |
|---|---|---|---|---|---|
| Baseline (wt=0.1) | 3.95 | NaN | 0.95 | **1.8%** (collapsed) | ~18 / 0.03 |
| lowlr (lr/2, wt=0.1) | 8.09 | 3.09 | 3.08 | **4.75%** (2/3 collapse) | ~82-148 |
| **lowwt (wt/10=0.01)** | 13.91 | 13.62 | 13.52 | **13.68%** ⭐ | ~137 / 0.23 |

**Conclusion**: ambient SIGReg collapse on TinyIN is a **regularizer-strength issue, not a learning-rate issue**. wt=0.01 (the LeJEPA paper's preferred setting for this scale) gives clean, consistent, 3/3-seed stability at ~13.7% probe. This rescues ambient SIGReg from "catastrophic failure" to a plausible baseline — though still lower than B+SIG+proj (14.0% @ 300ep, 16.77% @ 600ep). Paper implication: ambient SIGReg *works* at TinyIN scale with correct hparams; the Poincaré tangent geometry provides additional lift beyond just getting ambient right.

## A+VIC 600ep — NEGATIVE SCALING CONFIRMED (3/3, 2026-04-23)

| Seed | 300ep | 600ep | Δ |
|---|---|---|---|
| s42 | 8.64 | 4.76 | −3.88 |
| s123 | 6.58 | 5.86 | −0.72 |
| s7 | 8.28 | 6.06 | −2.22 |
| **Mean** | **7.8%** | **5.56%** | **−2.24pp** |

Ranks stay healthy (~161) — not collapse, genuine feature-quality degradation with more training. Model A regresses on all 3 seeds, Model B improves on all 3 seeds. **Paper-worthy negative finding**: vanilla Euclidean JEPA has negative scaling on TinyIN with ViT-Tiny. Strengthens the "B > A" claim (B not only wins, A actively gets worse).

## Open items / next experiments

- **No-projector TinyIN runs for B+SIGReg** — we have proj-only numbers. Need to confirm tangent-SIGReg also works without the BN projector, or whether projector is now crucial at scale.
- **Larger model / ImageNet-1k** to confirm the scaling curve extends beyond TinyIN.
- ~~**Deeper investigation of why ambient SIGReg collapses on TinyIN**~~ — **RESOLVED**: regularizer weight, not LR. wt=0.01 fully stabilizes (13.68% mean, 3/3).
- **Tangent probe numbers** are now available for all 36 runs (stored in results.json); most match euclidean probe within 0.5pp.

## Operational notes from this run

- 36 jobs, ran from ~22:00 Apr 21 to ~22:24 Apr 22 (~24 h wallclock).
- 3 jobs required resubmission due to watgpu1008 (no torch) or watgpu408 oversubscription.
- c100_b_sigreg_s123 was preempted 3 times before finally running on watgpu308.
- Empirical times: C10 ~9-10min, C100 ~35-40min, TinyIN ~3.5-4h per seed on healthy node.
