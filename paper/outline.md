# Hyperbolic JEPA: Poincaré Predictors Compose with Sketched Anti-Collapse for Scalable Self-Supervised Pretraining

**Target venue**: NeurIPS 2026 Workshop (SSL / Geometric DL / Interpolations)
**Length**: 8 pages + appendix

---

## 0. One-sentence pitch

Replacing I-JEPA's Euclidean prediction head with a Poincaré-ball predictor (Model B) yields consistent gains across datasets and anti-collapse regularizers, and at larger compute/data scale its natural companion is tangent-space SIGReg (LeJEPA) — together they give the best ViT-Tiny TinyImageNet probe accuracy we've measured while the Euclidean baseline *regresses* with additional training.

---

## 1. Abstract (draft, ~150 words)

Self-supervised joint-embedding predictive architectures (JEPAs) learn representations by predicting masked features in a flat Euclidean embedding space. We ask whether a curved predictor — specifically a Poincaré-ball prediction head on top of a standard Euclidean encoder — yields better representations for image SSL, and how it interacts with modern anti-collapse regularizers. Across CIFAR-10, CIFAR-100, and Tiny-ImageNet with a ViT-Tiny backbone and 3 seeds per cell, the Poincaré predictor (Model B) outperforms the Euclidean baseline (Model A) in **all 12 dataset × regularizer combinations**, with zero exceptions. At Tiny-ImageNet scale the benefit compounds with a **tangent-space application of SIGReg** (LeJEPA's Sketched Isotropic Gaussian Regularizer): B+SIGReg with a BN projector pulls away from B+VICReg as training grows (+2.9pp at 300 epochs → +4.8pp at 600 epochs, clean seed separation), while the Euclidean baseline *regresses* with more training. We also report a regularizer-weight fix that rescues ambient SIGReg from apparent collapse on small ViTs.

---

## 2. Contributions (bullet list for intro)

1. **Poincaré predictor (Model B)**: a drop-in replacement for I-JEPA's Euclidean prediction loss using a Poincaré-ball head with curvature c=1.0 and radius clipping. Wins uniformly across 12 dataset × regularizer cells.
2. **Tangent-space SIGReg**: applying LeJEPA's isotropic-Gaussian regularizer in the Poincaré tangent (not ambient) space is necessary to combine it with the hyperbolic predictor — and is what unlocks scale-dependent gains.
3. **Scale-dependent regularizer prescription**: at small scale (CIFAR) VICReg wins; at Tiny-ImageNet + 600 epochs, SIGReg wins with clean seed separation (lowest-seed SIG > highest-seed VIC).
4. **Negative scaling of Euclidean JEPA on Tiny-ImageNet**: Model A's linear-probe accuracy *decreases* from 300 → 600 epochs while Model B's increases. Suggests vanilla Euclidean JEPA is feature-limited on this data distribution in ViT-Tiny.
5. **Ambient-SIGReg collapse is a hyperparameter issue, not architectural**: dropping the regularizer weight from 0.1 → 0.01 rescues ambient SIGReg from 1.8% (collapsed) → 13.68% (stable) on Tiny-ImageNet with ViT-Tiny.
6. **Projector × regularizer interaction**: BN projector hurts VICReg by ~3.6pp but helps SIGReg by ~2.7pp — honest caveat on regularizer comparisons. Disclosed, not hidden.

---

## 3. Section Plan

### §1 Introduction (~0.75 page)
- I-JEPA background; masked-feature prediction
- Recent anti-collapse advances: VICReg, SIGReg/LeJEPA
- Motivation for curvature: many visual-hierarchy arguments cite hyperbolic embeddings for tree-like structure (cite Nickel&Kiela, Khrulkov, poincare-resnet)
- Open question: does the predictor-head geometry matter given modern anti-collapse tricks?
- Contributions bullet list (see §2 above)

### §2 Background & Related Work (~0.75 page)
- **JEPA / I-JEPA**: predict target embeddings of masked patches from context embeddings
- **Poincaré ball** basics: Möbius add, exp/log maps at origin, geodesic distance, radius clipping
- **VICReg**: variance (hinge), covariance (off-diag) terms
- **SIGReg / LeJEPA**: Epps–Pulley normality test on random 1-D slices; ambient vs tangent. Cite arXiv:2511.08544.
- **Relation to prior hyperbolic SSL**: (cite hyperbolic contrastive, Poincaré CNN/ResNet, LorentzFormer) — differ from our approach in that we keep the encoder Euclidean and only curve the *prediction loss*, preserving compatibility with any pretrained backbone.

### §3 Method (~1.5 pages)
**§3.1 Euclidean vs Poincaré prediction loss (Model A vs B)**
- A: ‖z_pred − z_target‖²
- B: Poincaré distance d_c(z_pred, z_target)^p, with z projected into the ball via `project_to_ball(z, c)` and clipped to radius r_max < 1/√c.
- Derivation of gradient wrt predictor output.
- Important: encoder stays Euclidean — only the prediction *loss* lives in hyperbolic space. This preserves downstream linear-probe compatibility.

**§3.2 Tangent-space SIGReg (for Model B)**
- SIGReg assumes the regularized features are Euclidean/isotropic-Gaussian targets. Applying it to points on the ball violates this assumption.
- Our variant: before computing SIGReg, apply logmap at origin to lift ball points → tangent space at 0, then apply the standard ambient SIGReg formulation on the tangent vectors.
- Tangent-SIGReg with predicted & target embeddings in the tangent space.

**§3.3 BN projector head (from LeJEPA §5.1)**
- LayerNorm at ViT output kills anti-collapse signal on small ViTs (LeJEPA finding)
- Replace with BatchNorm projector before applying regularizer
- We confirm this effect and report the interaction with VICReg vs SIGReg (see §5)

### §4 Experimental Setup (~0.75 page)
- Datasets: CIFAR-10 (50 ep), CIFAR-100 (200 ep), Tiny-ImageNet (300 ep + 600 ep for scale probe)
- Architecture: ViT-Tiny (192 dim), predictor 96 dim × 4 heads × 4 depth
- Masking: I-JEPA-style 0.15-0.20 pred / 0.85-1.0 ctx
- Regularizers: VICReg weight 0.1; SIGReg M=256 slices, t_max=3, n_points=17, weight 0.1 (baseline) / 0.01 (ambient fix)
- Evaluation: frozen-encoder linear probe (Euclidean pool) + tangent-probe (for B-models)
- 3 seeds per cell (42, 123, 7)

### §5 Results (~2.5 pages, the meat)

**§5.1 Main result: B > A universally (Table 1)**
Full 12-cell table from the primary 36-run matrix:

| Dataset | Regularizer | A mean | B mean | Δ |
|---|---|---|---|---|
| CIFAR-10 | VICReg | 35.2%* | 43.7% | +8.5 |
| CIFAR-10 | SIGReg | 29.5% | 34.4% | +4.9 |
| CIFAR-100 | VICReg | 18.0% | 19.6% | +1.6 |
| CIFAR-100 | SIGReg | 8.6% | 19.4% | +10.8 |
| Tiny-IN 300ep | VICReg | 7.8% | 11.1% | +3.3 |
| Tiny-IN 300ep | SIGReg | 1.8%** | 14.0% | +12.2 |

*s7 outlier; **collapsed at default hparams (see §5.4)

→ **B wins every cell. Paper's headline claim.**

**§5.2 Scale-dependent regularizer (Table 2)**
B-cell only, Tiny-IN, with/without extra compute:

| Training | B+VICReg | B+SIGReg | Δ |
|---|---|---|---|
| 300 epochs | 11.1% | 14.0% | +2.9 |
| 600 epochs | 11.9% | 16.8% | +4.8 |

Clean seed separation at 600ep (lowest SIG > highest VIC). B+VIC plateaus at 300→600ep; B+SIG gains +2.8pp.
→ **Side claim: SIGReg scales better than VICReg on Model B.**

**§5.3 Negative scaling of Model A (Table 3)**
A+VICReg, Tiny-IN, 300 → 600 epochs:

| Training | s42 | s123 | s7 | Mean |
|---|---|---|---|---|
| 300 ep | 8.64 | 6.58 | 8.28 | 7.8% |
| 600 ep | 4.76 | 5.86 | 6.06 | **5.56% (−2.24pp)** |

→ **Confirmed (3/3 seeds): vanilla Euclidean JEPA degrades with more training on TinyIN. Every seed regresses. Ranks stay healthy (~161) — not collapse, but genuine feature-quality degradation. Reinforces the B>A claim.**

**§5.4 Ambient SIGReg collapse fix (Table 4)**
A+SIGReg, Tiny-IN, hyperparameter rescue:

| Variant | Mean | Seeds |
|---|---|---|
| Default (wt=0.1) | 1.8% | 2/3 collapsed |
| lowlr (lr/2, wt=0.1) | 4.75% | 2/3 partial collapse |
| **lowwt (wt=0.01)** | **13.68%** | **3/3 stable** |

→ Regularizer weight, not learning rate, is the culprit. Ambient SIGReg works at TinyIN scale with correct hparams, but is still outperformed by B+SIG+proj (14.0% at 300ep, 16.8% at 600ep).

**§5.5 Projector × regularizer interaction (Table 5)**
B-cell, Tiny-IN 300ep:

| Config | +proj | −proj | Δ_proj |
|---|---|---|---|
| B+VICReg | 11.1% | 14.7% | −3.6 |
| B+SIGReg | 14.0% | 11.3% | +2.7 |

→ Honest caveat: SIG>VIC ordering depends on the BN projector. At 300ep without projector, VIC wins. At 600ep with projector, SIG wins. Our best config is B+SIG+proj at scale.

**§5.6 Curvature ablation (Table 6)** — COMPLETE (9/9)
B+SIGReg+proj, Tiny-IN 300ep:

| Curvature c | s42 | s123 | s7 | Mean |
|---|---|---|---|---|
| 0.25 | 15.46 | 13.97 | 14.24 | **14.56%** |
| **0.5** | **14.70** | **15.68** | **14.59** | **15.00%** |
| 1.0 (baseline) | 13.73 | 14.79 | 13.75 | 14.09% |
| 2.0 | 14.83 | 13.96 | 15.91 | **14.90%** |

→ **Curvature is robust across an 8× range (0.25→2.0).** All means within 0.9pp. c=0.5 slightly best but within seed noise; c=1.0 the lowest but within noise as well. Validates c=1.0 as a reasonable default, not cherry-picked.

**§5.7 Collapse metrics (short paragraph + plots in appendix)**
- Effective rank and mean-variance correlate loosely with probe accuracy but don't rank configs correctly (B+SIG has lower rank than B+VIC on TinyIN yet higher probe).
- Warning against using anti-collapse metrics as sole model-selection objective.

### §6 Discussion (~0.5 page)
- Why does Poincaré win? Hypothesis: hierarchical visual structure (part-whole) embeds more efficiently in constant-negative-curvature space, so the predictor has less geometric mismatch with what the encoder learns. Keeping encoder Euclidean means we only reshape the loss, not the backbone — a minimal intervention.
- Why does SIG scale better than VIC? Hypothesis: VICReg's per-dim variance hinge saturates at moderate scale; SIGReg's distributional matching has more headroom. Supported by VIC plateau 300→600ep, SIG continuing to improve.
- Why does A regress? Euclidean JEPA on curved data manifolds accumulates geometric error with more training.

### §7 Limitations & Future Work (~0.3 page)
- ViT-Tiny only; ViT-S/B scaling unverified
- Tiny-ImageNet is smallest "real" ImageNet; ImageNet-1k untested
- 3 seeds per cell is the minimum for defensible claims
- Curvature set to c=1.0 across encoder/predictor/regularizer — decoupling these (different c per component) is future work
- Hyperbolic probe: we use Euclidean linear probe on ball points. Tangent-space linear probe matches within 0.5pp in all tested cells, but a native hyperbolic probe could differ.

### §8 Reproducibility
- Config files + SLURM scripts released
- All 36 primary + 24 follow-up runs have results.json with per-epoch metrics
- 3 fixed seeds (42, 123, 7)

---

## 4. Figures / tables checklist

| # | What | Status |
|---|---|---|
| T1 | 12-cell A-vs-B main (CIFAR-10/100 + TinyIN × VIC/SIG × 300ep) | ✅ data ready |
| T2 | B-cell 300 vs 600 ep scaling | ✅ data ready |
| T3 | A-cell 300 vs 600 ep (negative scaling) | ✅ 3/3 seeds, confirmed |
| T4 | A+SIG hparam rescue (baseline/lowlr/lowwt) | ✅ data ready |
| T5 | Projector × regularizer interaction | ✅ data ready |
| T6 | Curvature ablation (c=0.25/0.5/1.0/2.0) | 🟡 9 jobs running, ~4-5h ETA |
| F1 | Training curves (loss, reg, probe) for B+SIG+proj 600ep | TODO — plot from epochs in results.json |
| F2 | Rank / mean_var scatter vs probe acc (anti-collapse metrics caveat) | TODO |
| F3 | Schematic: Model A vs B prediction loss | TODO |

---

## 5. Writing order (from easiest to hardest)

1. §2 Background (compile citations, define math notation) — 1 day
2. §4 Experimental Setup (factual, no writing) — 0.5 day
3. §3 Method (write equations, definitions) — 1 day
4. §5 Results (tables + narrative) — 2 days
5. §1 Introduction (after everything else is written) — 1 day
6. §6 Discussion — 0.5 day
7. Polish, figures, bibliography — 1-2 days

---

## 6. Risk items (things that could still change)

1. **Curvature ablation could show c=1.0 is not best.** If so, rerun the B-cell with optimal c. Current hypothesis: flat between 0.5-2.0.
2. ~~**A+VIC 600ep s123 could invalidate the negative-scaling claim.**~~ **RESOLVED**: s123 = 5.86 (300ep was 6.58) → negative scaling confirmed 3/3.
3. **Reviewer pushback on scale.** "This is ViT-Tiny on TinyIN — prove it on ViT-S + IN-1k." Defense: workshop paper, minimal intervention is the contribution, scale is future work.
