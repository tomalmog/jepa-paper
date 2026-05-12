# Tier 1.5 Results

Completed on WATGPU by 2026-05-12. All submitted jobs finished with SLURM
state `COMPLETED` and exit code `0:0`.

## Raw Results

| Cell | Seed | Job ID | Probe Euclidean | Probe Tangent | Rank | Mean Var | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| A+VICReg 600ep ViT-S | 42 | 1437014 | 10.35 | n/a | 283.03 | 0.6153 | stable |
| A+VICReg 600ep ViT-S | 123 | 1437016 | 0.50 | n/a | NaN | NaN | training produced NaNs |
| A+VICReg 600ep ViT-S | 7 | 1437017 | 0.50 | n/a | NaN | NaN | training produced NaNs |
| A+SIGReg 600ep ViT-S | 42 | 1437019 | 18.06 | n/a | 189.31 | 0.4945 | stable |
| A+SIGReg 600ep ViT-S | 123 | 1437021 | 16.41 | n/a | 188.10 | 0.5286 | stable |
| A+SIGReg 600ep ViT-S | 7 | 1437023 | 19.50 | n/a | 188.45 | 0.5295 | stable |
| B+SIGReg 600ep ViT-S | 314 | 1437025 | 15.50 | 15.44 | 108.40 | 0.6733 | stable |
| B+SIGReg 600ep ViT-S | 2718 | 1437026 | 15.16 | 15.18 | 114.47 | 0.6686 | stable |
| B+VICReg 600ep ViT-S | 314 | 1437028 | 13.86 | 13.88 | 230.86 | 0.2537 | stable |
| B+VICReg 600ep ViT-S | 2718 | 1437030 | 13.78 | 13.81 | 231.81 | 0.2282 | stable |

## Aggregates

### New Tier 1.5 A-Cells

| Cell | Seeds | Mean Probe | Stability |
|---|---|---:|---|
| A+VICReg 600ep ViT-S | 42, 123, 7 | 3.78 | 2/3 NaN-collapse; stable-only = 10.35 |
| A+SIGReg 600ep ViT-S | 42, 123, 7 | 17.99 | 3/3 stable |

### Updated 5-Seed B-Cells

Old Tier 1 B-cell seeds were 42, 123, and 7. Tier 1.5 added seeds 314 and
2718.

| Cell | Seed Values | 5-Seed Mean |
|---|---|---:|
| B+SIGReg 600ep ViT-S | 18.12, 14.28, 17.35, 15.50, 15.16 | 16.08 |
| B+VICReg 600ep ViT-S | 14.84, 16.07, 11.72, 13.86, 13.78 | 14.05 |

## Paper Implications

- The B+SIGReg-over-B+VICReg long-training result still holds with 5 seeds:
  16.08 vs 14.05.
- The long-training B-vs-A SIGReg claim **does not hold on ViT-S**:
  A+SIGReg 600ep reaches 17.99, beating B+SIGReg 600ep's 16.08.
- A+VICReg 600ep is unstable on ViT-S: 2/3 seeds produce NaNs and collapse to
  chance-level 0.50% probe. This is stronger than the previous "A+VICReg
  regresses" framing; it is now a long-training instability/collapse result.
- The paper should be reframed away from "Poincare predictor is best at scale"
  and toward: prediction geometry changes the regularizer/training dynamics,
  B+SIGReg is more stable/better than B+VICReg under long training, but ambient
  SIGReg with the right weight is the strongest ViT-S 600ep baseline in this
  matrix.
- Main-conference viability now depends even more on ImageNet-100 or a sharper
  mechanistic/conceptual story. As-is, this remains a good workshop paper, but
  the main claim must be rewritten carefully.
