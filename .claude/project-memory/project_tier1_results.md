---
name: Tier 1 results — COMPLETE (1A+1B+1C all done, 2026-04-30)
description: Tier 1 linear-probe results — B+VICReg REGRESSES from 300→600ep on ViT-S (16.55→14.21); B+SIG IMPROVES (15.10→16.58); paper-significant scaling reversal
type: project
originSessionId: 7f294af5-ddf3-4265-925e-bd7bd9f8a494
---
## 1A — TinyIN ViT-Tiny B+SIGReg @ 600ep, NO projector (3/3 complete)

| Seed | Probe Euc | Probe Tan |
|---|---|---|
| 42  | 11.83 | 11.82 |
| 123 | 12.97 | 13.02 |
| 7   | 12.95 | 12.86 |
| **Mean** | **12.58%** | **12.57%** |

**Comparison to prior runs (no projector):**
- B+VICReg+noproj @ 300ep ViT-Tiny: 14.70% — already in prior memory
- B+SIGReg+noproj @ 600ep ViT-Tiny: **12.58%** ← this run

**Conclusion:** Without projector, B+VICReg at 300ep STILL beats B+SIGReg at 600ep. The "B+SIG > B+VIC at TinyIN scale" finding from prior matrix is **projector-dependent** — confirms the projector × regularizer interaction is essential to SIGReg's win. Projector confound closed in the paper's favor: paper should state explicitly that B+SIGReg's win is conditional on projector.

## 1B — TinyIN ViT-Small @ 300ep + projector (12/12 COMPLETE)

| Cell | s42 | s123 | s7 | Mean |
|---|---|---|---|---|
| A+VICReg | 12.28 | 13.40 | 13.19 | **12.96%** |
| A+SIGReg | 16.06 | 15.35 | 16.56 | **15.99%** |
| B+VICReg | 16.33 | 16.29 | 17.03 | **16.55%** ⭐ |
| B+SIGReg | 15.74 | 13.82 | 15.75 | **15.10%** |

**Critical findings vs ViT-Tiny baselines:**

1. **B+VICReg wins on ViT-S 300ep** (16.55%), not B+SIGReg (15.10%). This **flips the ViT-Tiny headline** where B+SIGReg was the winner.
2. **A+SIGReg no longer collapses at ViT-S** (15.99%) — ambient SIGReg works fine on the larger model (was catastrophic at ViT-Tiny: 1.8% mean). Direct empirical support for LeJEPA's "needs scale" claim.
3. **B-vs-A Poincaré claim holds for VICReg cell** (B+VIC 16.55 vs A+VIC 12.96 = +3.6pp). It **fails for SIGReg cell** at 300ep (B+SIG 15.10 vs A+SIG 15.99 = -0.9pp).
4. **Tier 1 1C is the make-or-break for the existing paper headline.** Update 2026-04-30: B+SIG @ 600ep mean 16.58% DOES overtake B+SIG @ 300ep (15.10%) AND beats A+SIG @ 300ep (15.99%). The original paper claim survives — but only after ViT-S extension to 600ep. B+VIC @ 600ep still in flux (s42=14.84% is anomalous low).

## 1C — TinyIN ViT-Small @ 600ep + projector (6/6 COMPLETE)

| Cell | s42 | s123 | s7 | Mean | vs 300ep |
|---|---|---|---|---|---|
| B+SIGReg @ 600ep | 18.12 | 14.28 | 17.35 | **16.58%** | +1.48pp ↑ |
| B+VICReg @ 600ep | 14.84 | 16.07 | **11.72** | **14.21%** | **−2.34pp ↓** |

**Headline result: doubling training time on ViT-S inverts the regularizer ranking.**
- At 300ep: B+VIC (16.55) > B+SIG (15.10).
- At 600ep: B+SIG (16.58) > B+VIC (14.21).

B+SIG improved by +1.48pp (15.10 → 16.58), recovering and slightly exceeding B+VIC's 300ep peak. B+VIC REGRESSED by 2.34pp (16.55 → 14.21). Two of three B+VIC seeds (s7=11.72, s42=14.84) collapsed below the 1B equivalents. This is not flakiness — it's consistent.

**Interpretation candidates:**
1. B+VICReg overfits / collapses on ViT-S TinyIN past ~300ep — projector + VICReg has no temperature to slow representation collapse with more training.
2. B+SIGReg's tangent regularizer continues to provide useful signal at longer training — Sliced Isotropic Gaussian Regularization is a *better long-training regularizer* in the projector regime.
3. B+VIC s7 = 11.72% is an extreme outlier; need additional seeds to confirm.

**Variance is high on both cells.** B+VIC range = 4.35pp (11.72→16.07), B+SIG range = 3.84pp (14.28→18.12). 5 seeds would be needed to publish these means with confidence intervals.

**This makes the original paper headline (B+SIGReg+proj is the winning recipe) more defensible**, but only with the qualifier "at sufficient training length on ViT-S". The 1B 300ep snapshot would have looked like B+VIC wins; only by going to 600ep does the SIGReg story emerge.

**Disk-quota incident (2026-04-29):** Original 1C jobs 1432452/53/54/55 all FAILED simultaneously — `OSError: [Errno 122] Disk quota exceeded`. The `/u401/talmog` partition hit 100% (1.1T full, dominated by 602G `specialist-swarm` from another project; my Tier 1 runs alone were 95G with `save_every: 50` writing 12 checkpoints × 370MB per ViT-S run).

**Fix:** Edited 1C configs to `save_every: 600` (only `last.pt` overwritten + one `epoch0600.pt` final). Each 1C run now ~700MB instead of ~9G. Resubmitted as 1432529/30/31/32. 61G free is enough.

## Why these matter (final, post-1C)

These are **paper-changing** results. The current main.tex:
- Claims B+SIG+proj is the headline best on TinyIN — TRUE on ViT-Tiny, TRUE on ViT-S 600ep, FALSE on ViT-S 300ep.
- Claims Poincaré head wins universally — partially fails on ViT-S 300ep (A+SIG beats B+SIG).
- The A+SIG "catastrophic at scale" claim is now nuanced: catastrophic only at small-model scale.
- NEW finding: B+VICReg actually REGRESSES from 300→600ep on ViT-S. The "B+VIC is universally best" alternative reframing is now ruled out.

## How to apply

The paper's recipe (B+SIG+proj) survives, but the supporting story has to be rewritten:
1. ViT-S 300ep alone would have refuted the paper headline.
2. Need 600ep to recover the SIGReg story at this backbone — adds a "training-length matters" axis.
3. Worth running A+VICReg/A+SIGReg @ 600ep ViT-S to anchor B-vs-A claims at long training (currently we only have A+ at 300ep).
4. Worth adding 2 more seeds to both 1C cells (5 total) before quoting means in abstract.
5. Tier 2 IN-100 ViT-S should run at ≥300ep AND ≥600ep to confirm whether the regression repeats.

## Tier 1.5 follow-ups (recommended before Tier 2)

- **A+VICReg @ 600ep ViT-S × 3 seeds** (anchors B-vs-A at 600ep, ~90 GPU-h)
- **A+SIGReg @ 600ep ViT-S × 3 seeds** (anchors B-vs-A at 600ep for SIGReg cell, ~90 GPU-h)
- **B+SIGReg + B+VICReg @ 600ep ViT-S × 2 more seeds each** (n=5 for headline, ~120 GPU-h)
- Total Tier 1.5: ~12 jobs, ~300 GPU-h. Cheaper than Tier 2 and de-risks variance claims.

Cancelled jobs in this round: 1432423, 1432429, 1432431 (all CPU-fallback on watgpu308). Resubmits: 1432448 (=1432423), 1432450 (=1432429), 1432451 (=1432431).
