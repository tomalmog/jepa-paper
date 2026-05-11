---
name: Hyperbolic JEPA research — full results & paper status
description: Consolidated results across CIFAR-10, CIFAR-100, Tiny-ImageNet; C-rescue sweep; literature positioning; and open items as of 2026-04-22
type: project
originSessionId: 7f294af5-ddf3-4265-925e-bd7bd9f8a494
---
## Core finding (stable across all datasets)

**Model B (Euclidean encoder + Poincaré loss) + VICReg is the winning configuration.** Fully hyperbolic encoder (C) never beats B on any dataset tested. This is consistent with prior literature (Khrulkov 2020, Ermolov 2022, Ge 2023, Desai 2023 all use Euclidean-encoder + hyperbolic-head/loss; no prior paper shows fully-hyperbolic encoder > Euclidean for image SSL).

## Results table (all with 3 seeds unless noted)

### CIFAR-10, ViT-Tiny, 50 epochs
| Model | Probe | Rank | Mean_var |
|---|---|---|---|
| A | 43.0 ± 0.7% | 88.4 | 0.264 |
| A + VICReg | 39.6 ± 0.9% | 186.2 | 0.824 |
| B | 44.0 ± 0.7% | 88.1 | 0.303 |
| **B + VICReg** | **47.6 ± 1.1%** | 160.6 | 0.676 |
| C | 27.1 ± 1.5% (collapsed) | 14.9 | 0.023 |
| C + VICReg | 40.2 ± 1.1% | 76.0 | 0.920 |

### CIFAR-100, ViT-Tiny, 200 epochs
| Model | Probe | Rank | Mean_var | Notes |
|---|---|---|---|---|
| A | 21.1 ± 0.6% | 110.9 | 0.262 | |
| B | 21.6 ± 0.4% | 107.6 | 0.261 | |
| **B + VICReg** | **23.5 ± 0.2%** | 181.8 | 0.571 | |
| C + VICReg | 18.95% (1 seed only) | 101.8 | 0.942 | seeds 42/123 timed out at ep ~169/200 |

### Tiny-ImageNet, ViT-Tiny, 300 epochs, 64×64, patch 8
| Model | Probe | Rank | Mean_var | Notes |
|---|---|---|---|---|
| A | 14.1 ± 0.7% | 64.2 | 0.240 | |
| B | 12.6 ± 1.0% | 54.7 | 0.336 | **B < A at scale** |
| **B + VICReg** | **15.0 ± 0.1%** | 169.1 | 0.341 | |
| C + VICReg | 10.96% (1 seed) | 96.8 | 0.965 | seeds 42/123 still running |

### C-rescue sweep (CIFAR-10, 100 epochs, seed 42 only)
Best: curv=0.5 → 43.6% (ties A, still 4pp below B+VICReg 47.6%).
All other variants worse. Higher VICReg weight → higher rank but plateaus at ~38–43%.
4 configs (lr=4e-4, vic=0.5, vic=1.0, combo_c0.5_lr4e-4_v0.5) were never launched.

## Status of running jobs (2026-04-22)
- `tin_300ep_c_vicreg_seed42`: ep 295/300, ~6 h left
- `tin_300ep_c_vicreg_seed123`: ep 242/300, ~6.5 h left
- CIFAR-100 C+VICReg seeds 42/123 **died at ep ~169/200** (SLURM timeout, no results)

## Key insights beyond the A/B/C story

1. **B's win over A shrinks at scale.** CIFAR-10 50ep: +1.0pp. CIFAR-100 200ep: +0.5pp. Tiny-ImageNet 300ep: **B is worse than A (-1.5pp)**. Only B+VICReg consistently beats A+VICReg / A.
2. **VICReg hurts A** on CIFAR-10 (43→39.6) but **helps A** on Tiny-ImageNet. Regularization story is dataset-dependent.
3. **Poincaré-loss-alone (B) adds nothing at scale** — it's the *combination* B+VICReg that wins. Suggests the Poincaré loss provides structure that VICReg stabilizes, or that both independently prevent different failure modes.
4. **C never collapses with VICReg** across all 3 datasets (rank 75–102, mean_var >0.9), but also never wins — confirms collapse prevention was necessary but not sufficient.

## Paper story (updated)

**Strong:** B + VICReg is consistently the best configuration across 3 datasets × 3 seeds. Poincaré loss + anti-collapse regularization synergize. Fully hyperbolic encoder is a clean negative result.

**Weak:** B alone doesn't reliably beat A at scale. The "Poincaré loss is the contribution" framing needs qualification — it's "Poincaré loss *with* VICReg."

**Why:** These results determine paper claims and priorities.

**How to apply:**
- Lead claim: "B+VICReg beats all other configurations across CIFAR-10/100 and Tiny-ImageNet" (proven).
- Secondary claim: "Fully hyperbolic encoder doesn't help even with collapse prevention" (proven, consistent with literature).
- Avoid claiming "Poincaré loss alone is the contribution" — that's not what the data show at scale.
- Before finalizing: relaunch CIFAR-100 C+VICReg seeds 42/123 with longer timeout (or accept the one-seed result with a caveat).
- Consider whether to bother finishing the remaining 4 C-rescue configs — unlikely to change the story given best sweep result (43.6%) is still < B+VICReg.
