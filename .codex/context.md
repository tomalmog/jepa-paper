# Codex Context

Current handoff, created 2026-05-11.

This repository is a Hyperbolic-JEPA research and paper workspace. The latest
Claude context has been copied into `.claude/project-memory/` and the most
recent Claude JSONL conversation is in `.claude/conversations/`.

Important state:

- Tier 1 Tiny-ImageNet ViT-S experiments are complete.
- The live decision is whether to run Tier 1.5 before Tier 2 ImageNet-100.
- Tier 1.5 recommendation: A+VICReg and A+SIGReg at 600 epochs on ViT-S, plus
  two extra seeds for B+SIGReg and B+VICReg at 600 epochs.
- Current key result: B+SIGReg improves from 15.10% at 300 epochs to 16.58% at
  600 epochs on ViT-S Tiny-ImageNet, while B+VICReg regresses from 16.55% to
  14.21%.
- Tier 2 ImageNet-100 is blocked on dataset preparation, class-list choice, and
  tight disk space on the cluster.
- Cluster scripts should exclude `watgpu1008`, `watgpu408`, and `watgpu308`;
  some older local scripts only exclude the first two.
- ViT-S 600-epoch configs should use sparse checkpointing (`save_every: 600`)
  on the cluster to avoid quota failures; some local configs still say
  `save_every: 50`.

Paper state:

- The project is a credible workshop paper now.
- It is not yet a strong main-conference submission without scale validation,
  preferably ImageNet-100 or larger.
- The framing should emphasize conditional scaling behavior and regularizer /
  projector interactions, not an unconditional "hyperbolic wins universally"
  claim.
