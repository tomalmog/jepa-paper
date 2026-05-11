---
name: Tier 1 experiment matrix launched (2026-04-28)
description: 21-job Tier 1 matrix submitted to watgpu — closes projector-confound gap and adds ViT-Small scaling story
type: project
originSessionId: 7f294af5-ddf3-4265-925e-bd7bd9f8a494
---
Tier 1 submitted 2026-04-28 via `scripts/submit_tier1.sh` (queue-gated launcher, max_pending=10, max_total=18). Total ~340 GPU-h, ~5–7 days elapsed.

**Composition (21 jobs, 3 seeds each: 42, 123, 7):**
- **1A (3 jobs, 14h walltime each):** `tin_600ep_b_sigreg_noproj` — ViT-Tiny B+SIGReg @ 600ep without BN projector. Closes the self-acknowledged confound: does SIGReg only help when paired with the projector, or does the regularizer still help on its own?
- **1B (12 jobs, 22h walltime each):** ViT-Small TinyIN 300ep, full 2×2 matrix `{A,B}×{vicreg,sigreg}+proj`. Replicates the ViT-Tiny headline matrix on a larger backbone — establishes the scaling story.
- **1C (6 jobs, 40h walltime each):** ViT-Small TinyIN 600ep B-cell only `{vicreg, sigreg}+proj`. Extends 1B to longer training; the cells where SIGReg expected to overtake VICReg.

**ViT-Small spec:** `embed_dim=384, depth=12, num_heads=6` (factory at `vision_transformer.py:329-331`). Predictor scaled to `192/4/4` to preserve ~0.5 dim ratio.

**Initial state (just-launched check):**
- Healthy: 6 running jobs reached epoch 6–7 within ~5 min, losses well-behaved (B+SIG-noproj ~5.6, B+SIG-proj-vits ~7.6, no NaN).
- Co-residency observed on watgpu208/308/508 (2 jobs/node). Per cluster reference, watch for stalled progress; not yet a problem.
- Smoke test (`/tmp/smoke_vits.yaml`, job 1432422) was used to verify ViT-S wiring before the matrix; it ran cleanly to ~iter 180 then was cancelled.

**SLURM job IDs:** 1432423–1432443 (sequential — first 16 confirmed at submit time, last 5 will be submitted as queue gate opens).

**Why:** User has 4 months to NeurIPS-quality submission and generous compute. Tier 1 closes one self-acknowledged gap (1A) and extends the matrix to a second backbone (1B+1C). Tier 2 (ImageNet-100) and Tier 3 (ImageNet-1k smoke) are gated on Tier 1 results.

**How to apply:** When checking progress, use the persistent monitor (task `b5bn9g1g5`) for failures; SSH directly to inspect logs at `/u401/talmog/jepa-paper/logs/`. When all 21 jobs complete, gather final eval numbers and decide whether to launch Tier 2.
