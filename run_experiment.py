"""
Unified experiment runner: pretrain + collapse metrics + linear probe.

Usage:
    python run_experiment.py --config configs/experiments/model_a.yaml --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.datasets.cifar import build_eval_loaders, build_pretrain_loader
from src.datasets.imagenet import (
    build_eval_loaders_imagenet,
    build_pretrain_loader_imagenet,
)
from src.helper import init_models, init_optimizer
from src.masks.multiblock import MultiBlockMaskCollator, apply_masks
from src.models.hyperbolic import build_loss, clip_feature, logmap0
from src.sigreg import build_sigreg
from src.vicreg import build_vicreg
from src.utils.schedulers import (
    CosineWDSchedule,
    MomentumSchedule,
    WarmupCosineSchedule,
)
from src.utils.tensors import AverageMeter, update_ema


# -----------------------------------------------------------------------------
# Projector head for regularization
# -----------------------------------------------------------------------------
class BNProjector(torch.nn.Module):
    """1-layer MLP + BatchNorm projector before the regularizer.

    Needed because the ViT's final LayerNorm re-normalizes mean/var, which
    prevents anti-collapse regularizers (VICReg, SIGReg) from biting the
    encoder representation effectively. Flagged in LeJEPA Section 5.1.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)
        self.bn = torch.nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(x))


def build_projector(cfg: dict, dim: int, device: torch.device) -> torch.nn.Module | None:
    if not cfg.get("projector", False):
        return None
    return BNProjector(dim).to(device)


# -----------------------------------------------------------------------------
# Device / precision helpers
# -----------------------------------------------------------------------------
def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp_ok(device: torch.device, amp_cfg: bool) -> bool:
    return bool(amp_cfg) and device.type == "cuda"


class _NoOpScaler:
    def scale(self, loss): return loss
    def unscale_(self, opt): pass
    def step(self, opt): opt.step()
    def update(self): pass


def _make_scaler(use_amp: bool):
    if use_amp:
        return torch.amp.GradScaler("cuda", enabled=True)
    return _NoOpScaler()


def _autocast_ctx(use_amp: bool):
    if use_amp:
        return torch.amp.autocast("cuda", enabled=True)

    class _NullCtx:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _NullCtx()


# -----------------------------------------------------------------------------
# Collapse metrics
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_collapse_metrics(encoder, loader, device, max_batches=20):
    """Compute feature-level collapse diagnostics."""
    encoder.eval()
    feats = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        h = encoder(x)  # (B, N, D)
        h = F.layer_norm(h, (h.size(-1),))
        h = h.mean(dim=1)  # global pool -> (B, D)
        feats.append(h)
    feats = torch.cat(feats, dim=0)  # (N_total, D)

    # Per-dimension variance
    var_per_dim = feats.var(dim=0)  # (D,)
    mean_var = var_per_dim.mean().item()
    min_var = var_per_dim.min().item()
    max_var = var_per_dim.max().item()

    # Effective rank (via singular values of centered features)
    centered = feats - feats.mean(dim=0, keepdim=True)
    # Use SVD on a manageable subset
    if centered.size(0) > 2048:
        centered = centered[:2048]
    try:
        s = torch.linalg.svdvals(centered)
        # Normalize singular values to form a distribution
        p = s / s.sum()
        # Shannon entropy -> effective rank = exp(entropy)
        entropy = -(p * (p + 1e-10).log()).sum().item()
        effective_rank = np.exp(entropy)
    except Exception:
        effective_rank = -1.0

    # Uniformity (Wang & Isola 2020) — how spread out on the hypersphere
    feats_normed = F.normalize(feats[:1024], dim=1)
    pdist = torch.cdist(feats_normed, feats_normed, p=2)
    # uniformity = log(mean(exp(-2 * ||z_i - z_j||^2)))
    uniformity = torch.exp(-2 * pdist.pow(2)).mean().log().item()

    encoder.train()
    return {
        "mean_var": mean_var,
        "min_var": min_var,
        "max_var": max_var,
        "effective_rank": effective_rank,
        "uniformity": uniformity,
    }


# -----------------------------------------------------------------------------
# Target features (with optional L2 normalization for confounder test)
# -----------------------------------------------------------------------------
def _target_features(
    target_encoder: torch.nn.Module,
    images: torch.Tensor,
    masks_pred: list[torch.Tensor],
    normalize: bool = False,
) -> torch.Tensor:
    with torch.no_grad():
        h = target_encoder(images)
        h = F.layer_norm(h, (h.size(-1),))
        if normalize:
            h = F.normalize(h, dim=-1)
        gathered = apply_masks(h, masks_pred)
    return gathered


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train_one_epoch(
    epoch, cfg, loader, context_encoder, target_encoder, predictor,
    optimizer, lr_sched, wd_sched, mom_sched, loss_fn, scaler, device, use_amp,
    normalize_targets=False,
    vicreg_fn=None, vicreg_weight=1.0,
    sigreg_fn=None, sigreg_weight=0.1,
    projector=None,
):
    context_encoder.train()
    predictor.train()
    target_encoder.eval()
    if projector is not None:
        projector.train()

    loss_meter = AverageMeter()
    reg_meter = AverageMeter()
    norm_meter = AverageMeter()
    tic = time.time()

    for it, (images, masks_enc, masks_pred) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
        masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

        lr = lr_sched.step()
        wd = wd_sched.step()
        m = mom_sched.step()

        with _autocast_ctx(use_amp):
            target_repr = _target_features(
                target_encoder, images, masks_pred, normalize=normalize_targets
            )
            ctx_repr = context_encoder(images, masks=masks_enc)
            pred_repr = predictor(ctx_repr, masks_enc, masks_pred)

            if normalize_targets:
                pred_repr = F.normalize(pred_repr, dim=-1)

            loss = loss_fn(pred_repr, target_repr)
            reg_loss_val = 0.0

            # Anti-collapse regularizer (VICReg or SIGReg) on pooled ctx features
            if vicreg_fn is not None or sigreg_fn is not None:
                ctx_pooled = ctx_repr.mean(dim=1)  # (B, D)
                norm_meter.update(ctx_pooled.norm(dim=-1).mean().item(), n=ctx_pooled.size(0))
                z = projector(ctx_pooled) if projector is not None else ctx_pooled
                if vicreg_fn is not None:
                    reg_loss = vicreg_fn(z)
                    loss = loss + vicreg_weight * reg_loss
                    reg_loss_val = reg_loss.item()
                elif sigreg_fn is not None:
                    reg_loss = sigreg_fn(z)
                    loss = loss + sigreg_weight * reg_loss
                    reg_loss_val = reg_loss.item()
                reg_meter.update(reg_loss_val, n=images.size(0))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg["grad_clip"] is not None:
            if use_amp:
                scaler.unscale_(optimizer)
            clip_params = list(context_encoder.parameters()) + list(predictor.parameters())
            if projector is not None:
                clip_params += list(projector.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        update_ema(context_encoder, target_encoder, momentum=m)

        loss_meter.update(loss.item(), n=images.size(0))

        if it % cfg["log_every"] == 0:
            reg_str = f"  reg={reg_meter.avg:.4f}" if reg_meter.count > 0 else ""
            norm_str = f"  ‖z‖={norm_meter.avg:.3f}" if norm_meter.count > 0 else ""
            print(
                f"[epoch {epoch:3d} it {it:4d}/{len(loader)}] "
                f"loss={loss_meter.avg:.4f}{reg_str}{norm_str}  "
                f"lr={lr:.2e}  wd={wd:.2e}  ema={m:.4f}  "
                f"elapsed={time.time()-tic:.1f}s",
                flush=True,
            )

    return {
        "loss": loss_meter.avg,
        "reg_loss": reg_meter.avg if reg_meter.count > 0 else None,
        "ctx_norm": norm_meter.avg if norm_meter.count > 0 else None,
    }


# -----------------------------------------------------------------------------
# Linear probe
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_features(encoder, loader, device, curvature=None, clip_radius=None):
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        h = encoder(x)
        h = F.layer_norm(h, (h.size(-1),))
        h = h.mean(dim=1)
        if curvature is not None:
            h = clip_feature(h, clip_radius)
            h = logmap0(h, curvature)
        feats.append(h.cpu())
        labels.append(y)
    return torch.cat(feats, 0), torch.cat(labels, 0)


def run_linear_probe(encoder, cfg, device, probe_epochs=100, use_tangent=False):
    _ds = cfg["dataset"].lower()
    if _ds in ("cifar10", "cifar100"):
        train_loader, test_loader, n_classes = build_eval_loaders(
            root=cfg["data_root"], dataset=cfg["dataset"],
            batch_size=512, num_workers=cfg["num_workers"], img_size=cfg["img_size"],
        )
    else:
        train_loader, test_loader, n_classes = build_eval_loaders_imagenet(
            root=cfg["data_root"], dataset=cfg["dataset"],
            batch_size=512, num_workers=cfg["num_workers"], img_size=cfg["img_size"],
        )
    curv = cfg["curvature"] if use_tangent else None
    clip = cfg["clip_radius"] if use_tangent else None

    Xtr, ytr = extract_features(encoder, train_loader, device, curvature=curv, clip_radius=clip)
    Xte, yte = extract_features(encoder, test_loader, device, curvature=curv, clip_radius=clip)

    Xtr = F.normalize(Xtr, dim=1)
    Xte = F.normalize(Xte, dim=1)

    probe = torch.nn.Linear(Xtr.size(1), n_classes).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=probe_epochs)

    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xte, yte = Xte.to(device), yte.to(device)

    best_acc = 0.0
    for ep in range(probe_epochs):
        probe.train()
        perm = torch.randperm(Xtr.size(0), device=device)
        for i in range(0, Xtr.size(0), 512):
            idx = perm[i:i + 512]
            logits = probe(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        probe.eval()
        with torch.no_grad():
            acc = (probe(Xte).argmax(1) == yte).float().mean().item()
        best_acc = max(best_acc, acc)
        if (ep + 1) % 20 == 0:
            print(f"  [probe ep {ep+1:3d}] acc={acc*100:.2f}%  best={best_acc*100:.2f}%")

    return best_acc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Append seed to output dir
    seed = args.seed
    cfg["output_dir"] = f"{cfg['output_dir']}_seed{seed}"
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _pick_device()
    use_amp = _amp_ok(device, cfg.get("amp", False))
    normalize_targets = cfg.get("normalize_targets", False)
    print(f"Device: {device} | AMP: {use_amp} | Seed: {seed}")
    print(f"Config: {args.config}")
    print(f"Output: {out_dir}")
    print(f"Normalize targets: {normalize_targets}")

    # Dataloader + masking
    collator = MultiBlockMaskCollator(
        input_size=cfg["img_size"], patch_size=cfg["patch_size"],
        enc_mask_scale=tuple(cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(cfg["pred_mask_scale"]),
        aspect_ratio=tuple(cfg["aspect_ratio"]),
        num_enc_masks=cfg["num_enc_masks"],
        num_pred_masks=cfg["num_pred_masks"],
        min_keep=cfg["min_keep"],
        allow_overlap=cfg["allow_overlap"],
    )
    _ds = cfg["dataset"].lower()
    if _ds in ("cifar10", "cifar100"):
        loader, _ = build_pretrain_loader(
            root=cfg["data_root"], dataset=cfg["dataset"],
            batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
            img_size=cfg["img_size"], collate_fn=collator,
        )
    else:
        loader, _ = build_pretrain_loader_imagenet(
            root=cfg["data_root"], dataset=cfg["dataset"],
            batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
            img_size=cfg["img_size"], collate_fn=collator,
        )

    # Models
    context_encoder, target_encoder, predictor = init_models(
        encoder_name=cfg["encoder"], img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        predictor_embed_dim=cfg["predictor_embed_dim"],
        predictor_depth=cfg["predictor_depth"],
        predictor_num_heads=cfg["predictor_num_heads"],
        device=device, curvature=float(cfg.get("curvature", 1.0)),
    )
    # Build projector head (for anti-collapse regularizers). Dim inferred from encoder.
    enc_dim = context_encoder.embed_dim if hasattr(context_encoder, "embed_dim") else None
    if enc_dim is None:
        # Fallback: probe one forward pass
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, cfg["img_size"], cfg["img_size"], device=device)
            enc_dim = context_encoder(_dummy).size(-1)
    projector = build_projector(cfg, enc_dim, device)

    optimizer = init_optimizer(
        context_encoder, predictor, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    if projector is not None:
        # Add projector params to optimizer as an additional group
        optimizer.add_param_group({
            "params": list(projector.parameters()),
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
        })

    steps_per_epoch = len(loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    lr_sched = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        start_lr=cfg["start_lr"], ref_lr=cfg["lr"], final_lr=cfg["final_lr"],
    )
    wd_sched = CosineWDSchedule(
        optimizer, ref_wd=cfg["weight_decay"],
        final_wd=cfg["final_weight_decay"], total_steps=total_steps,
    )
    mom_sched = MomentumSchedule(
        ema_start=cfg["ema_start"], ema_end=cfg["ema_end"], total_steps=total_steps,
    )
    loss_fn = build_loss(cfg)
    vicreg_fn = build_vicreg(cfg)
    vicreg_weight = float(cfg.get("vicreg_weight", 1.0))
    sigreg_fn = build_sigreg(cfg)
    if sigreg_fn is not None:
        sigreg_fn = sigreg_fn.to(device)
    sigreg_weight = float(cfg.get("sigreg_weight", 0.1))
    scaler = _make_scaler(use_amp)

    if vicreg_fn is not None and sigreg_fn is not None:
        raise ValueError("Both VICReg and SIGReg are enabled — choose one.")

    print(f"Loss: {loss_fn.__class__.__name__}")
    if vicreg_fn is not None:
        print(f"VICReg: ON (weight={vicreg_weight}, var_w={vicreg_fn.var_weight}, cov_w={vicreg_fn.cov_weight})")
    elif sigreg_fn is not None:
        mode = cfg.get("sigreg_mode", "tangent")
        M = cfg.get("sigreg_num_slices", 256)
        print(f"SIGReg: ON (mode={mode}, weight={sigreg_weight}, M={M})")
    else:
        print("Regularizer: OFF")
    if projector is not None:
        print(f"Projector: BN (dim={enc_dim})")
    else:
        print("Projector: OFF")
    print(f"Encoder: {cfg['encoder']} | Params: {sum(p.numel() for p in context_encoder.parameters()):,}")
    print("=" * 60)

    # Build eval loader for collapse metrics
    if _ds in ("cifar10", "cifar100"):
        eval_train_loader, _, _ = build_eval_loaders(
            root=cfg["data_root"], dataset=cfg["dataset"],
            batch_size=256, num_workers=cfg["num_workers"], img_size=cfg["img_size"],
        )
    else:
        eval_train_loader, _, _ = build_eval_loaders_imagenet(
            root=cfg["data_root"], dataset=cfg["dataset"],
            batch_size=256, num_workers=cfg["num_workers"], img_size=cfg["img_size"],
        )

    # Train
    results = {"config": args.config, "seed": seed, "epochs": []}
    for epoch in range(cfg["epochs"]):
        epoch_stats = train_one_epoch(
            epoch=epoch, cfg=cfg, loader=loader,
            context_encoder=context_encoder, target_encoder=target_encoder,
            predictor=predictor, optimizer=optimizer,
            lr_sched=lr_sched, wd_sched=wd_sched, mom_sched=mom_sched,
            loss_fn=loss_fn, scaler=scaler, device=device, use_amp=use_amp,
            normalize_targets=normalize_targets,
            vicreg_fn=vicreg_fn, vicreg_weight=vicreg_weight,
            sigreg_fn=sigreg_fn, sigreg_weight=sigreg_weight,
            projector=projector,
        )
        avg_loss = epoch_stats["loss"]

        epoch_data = {"epoch": epoch, "loss": avg_loss}
        if epoch_stats["reg_loss"] is not None:
            epoch_data["reg_loss"] = epoch_stats["reg_loss"]
        if epoch_stats["ctx_norm"] is not None:
            epoch_data["ctx_norm"] = epoch_stats["ctx_norm"]

        # Collapse metrics every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == cfg["epochs"] - 1:
            metrics = compute_collapse_metrics(
                target_encoder, eval_train_loader, device
            )
            epoch_data["collapse_metrics"] = metrics
            print(f"  [collapse] rank={metrics['effective_rank']:.1f}  "
                  f"mean_var={metrics['mean_var']:.4f}  "
                  f"uniformity={metrics['uniformity']:.4f}")

        results["epochs"].append(epoch_data)

        # Checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "context_encoder": context_encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg, "loss": avg_loss,
        }
        if projector is not None:
            ckpt["projector"] = projector.state_dict()
        torch.save(ckpt, out_dir / "last.pt")
        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save(ckpt, out_dir / f"epoch{epoch+1:04d}.pt")

    # =========================================================================
    # Linear Probe
    # =========================================================================
    print("\n" + "=" * 60)
    print("LINEAR PROBE (Euclidean pool)")
    print("=" * 60)
    eucl_acc = run_linear_probe(target_encoder, cfg, device, use_tangent=False)
    results["probe_euclidean"] = eucl_acc
    print(f"\n>>> Euclidean probe accuracy: {eucl_acc*100:.2f}%")

    # Also run tangent probe if this is a hyperbolic model
    if cfg.get("loss_type", "poincare") == "poincare":
        print("\n" + "=" * 60)
        print("LINEAR PROBE (Tangent pool)")
        print("=" * 60)
        tang_acc = run_linear_probe(target_encoder, cfg, device, use_tangent=True)
        results["probe_tangent"] = tang_acc
        print(f"\n>>> Tangent probe accuracy: {tang_acc*100:.2f}%")

    # Final collapse metrics
    print("\n" + "=" * 60)
    print("FINAL COLLAPSE METRICS")
    print("=" * 60)
    final_metrics = compute_collapse_metrics(target_encoder, eval_train_loader, device)
    results["final_collapse_metrics"] = final_metrics
    print(json.dumps(final_metrics, indent=2))

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")
    print(f"DONE. Probe acc: {eucl_acc*100:.2f}%")


if __name__ == "__main__":
    main()
