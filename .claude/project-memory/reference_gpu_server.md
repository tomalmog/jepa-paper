---
name: GPU cluster (Waterloo watgpu)
description: SSH access, project paths, SLURM patterns, and known-bad nodes for the Waterloo cluster
type: reference
originSessionId: 7f294af5-ddf3-4265-925e-bd7bd9f8a494
---
## Current cluster: Waterloo watgpu

- **SSH**: `ssh talmog@watgpu.cs.uwaterloo.ca` (user-key auth, no password)
- **Project root on cluster**: `/u401/talmog/jepa-paper/`
- **Config dir**: `/u401/talmog/jepa-paper/configs/experiments/`
- **Run outputs**: `/u401/talmog/jepa-paper/runs/`
- **SLURM logs**: `/u401/talmog/jepa-paper/logs/`

## Node notes

- **watgpu1008 has no torch installed** in the system Python. ALWAYS add `#SBATCH --exclude=watgpu1008` to sbatch scripts, or jobs crash instantly with `ModuleNotFoundError: No module named 'torch'`.
- **watgpu408 gets oversubscribed easily** (SLURM sometimes packs 7+ jobs onto it, timesharing the GPU). If jobs on watgpu408 are progressing one epoch per 10+ minutes, they're stuck — kill and resubmit with `#SBATCH --exclude=watgpu1008,watgpu408`.
- **watgpu308 has broken CUDA init (2026-04-29)**. New jobs landing here get `Error 101: invalid device ordinal` and silently fall back to `Device: cpu | AMP: False`, then run ~50× slower than budget and time out. Already-running jobs on the node keep working — only fresh allocations hit the break. ALWAYS add `watgpu308` to the exclude list. If a job logs `Device: cpu` in its first few lines, it's hit this; cancel and resubmit excluding the node.
- **Reliable nodes**: watgpu208, 508, 608, 708 have all worked well in this round.
- Partition name: `ALL`

## SLURM header template

```bash
#SBATCH --job-name=<name>
#SBATCH --output=/u401/talmog/jepa-paper/logs/<name>_%j.out
#SBATCH --error=/u401/talmog/jepa-paper/logs/<name>_%j.err
#SBATCH --time=HH:MM:SS
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008
```

## Time budgets (empirical, seen on healthy nodes)

- CIFAR-10 50ep ViT-Tiny: ~9-10 min
- CIFAR-100 200ep ViT-Tiny: ~38-45 min
- Tiny-ImageNet 300ep ViT-Tiny (img 64, patch 8): ~4-6 h (longer if contested)
- Tiny-ImageNet 300ep ViT-Small (img 64, patch 8): ~12-15 h
- Tiny-ImageNet 600ep ViT-Small (img 64, patch 8): ~20-30 h (varies by node contention)

## Datasets present on cluster

- `~/data/cifar-10-batches-py/` and `~/data/cifar-100-python/`
- `~/data/tiny-imagenet-200/` (537M)
- **NOT YET PRESENT**: `~/data/imagenet-100/`. Tier 2 is gated on getting this. ImageNet-100 needs ~13GB and there's currently 42G free on /u401 (91% used). Choice of class list matters for reproducibility (multiple "imagenet-100" splits exist; standard reference is the one used by SimCLR/MoCo/SwAV — class list at `https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt`).

## Login-node caveat

The login node has `/usr/bin/python3` without torch — don't try `python3 -c "import torch"` on it. Compute nodes have torch globally (except watgpu1008).

## Old (stale) server — DO NOT USE

Previously used Vast.ai `ssh -p 32161 root@120.238.149.205` — this instance was stopped, retained here for history only.

## Why this matters

Two pre-flight checks that would have saved 30+ minutes earlier:
1. Submit a smoke-test job FIRST, verify the env + config, then fire the matrix.
2. Always add `--exclude=watgpu1008` to be safe.

## How to apply

Use this reference every time we submit jobs, check experiment progress, or encounter `ModuleNotFoundError` / inexplicably slow runs.
