# Tier 1.5 WATGPU Jobs

Submitted from this workspace on 2026-05-11/12 via
`scripts/submit_tier1_5.sh`.

Purpose:

- Anchor long-training ViT-S Model A baselines at 600 epochs.
- Add two more seeds to the high-variance B+SIGReg and B+VICReg 600-epoch
  headline cells.

Jobs:

| Job ID | Config | Seed | Initial state |
|---|---|---:|---|
| 1437014 | `tin_600ep_vits_a_vicreg_proj` | 42 | Running on `watgpu108`; confirmed `Device: cuda`, AMP on |
| 1437016 | `tin_600ep_vits_a_vicreg_proj` | 123 | Pending at submission |
| 1437017 | `tin_600ep_vits_a_vicreg_proj` | 7 | Pending at submission |
| 1437019 | `tin_600ep_vits_a_sigreg_proj` | 42 | Pending at submission |
| 1437021 | `tin_600ep_vits_a_sigreg_proj` | 123 | Pending at submission |
| 1437023 | `tin_600ep_vits_a_sigreg_proj` | 7 | Pending at submission |
| 1437025 | `tin_600ep_vits_b_sigreg_proj` | 314 | Pending at submission |
| 1437026 | `tin_600ep_vits_b_sigreg_proj` | 2718 | Pending at submission |
| 1437028 | `tin_600ep_vits_b_vicreg_proj` | 314 | Pending at submission |
| 1437030 | `tin_600ep_vits_b_vicreg_proj` | 2718 | Pending at submission |

Cluster checks:

- WATGPU free space before submission: 44 GB free on `/u401/talmog`.
- `tiny-imagenet-200` present at `/u401/talmog/data/tiny-imagenet-200`.
- Submitter excludes `watgpu1008`, `watgpu408`, and `watgpu308`.
- All ViT-S 600-epoch configs use `save_every: 600` to avoid quota blowups.

Monitor commands:

```bash
ssh talmog@watgpu.cs.uwaterloo.ca "squeue -u talmog -o '%.10i %.36j %.8T %.10M %.12R'"
ssh talmog@watgpu.cs.uwaterloo.ca "cd /u401/talmog/jepa-paper && tail -n 40 logs/tin_600ep_vits_*_14370*.out"
```
