"""
Hyperbolic-JEPA pretraining loop.

Mirrors the structure of ``src.train.main`` in facebookresearch/ijepa:

    context_repr = context_encoder(images, masks=masks_enc)
    target_repr  = target_encoder(images)                 # full image
    target_repr  = gather(target_repr, masks_pred)        # target patches

    pred_repr    = predictor(context_repr, masks_enc, masks_pred)

    loss = HYPERBOLIC_DISTANCE(pred_repr, target_repr)    # <-- only change

    backward; adamw step; EMA update of target encoder.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.datasets.cifar import build_pretrain_loader
from src.helper import init_models, init_optimizer
from src.masks.multiblock import MultiBlockMaskCollator, apply_masks
from src.models.hyperbolic import PoincareRegressionLoss, build_loss
from src.utils.schedulers import (
    CosineWDSchedule,
    MomentumSchedule,
    WarmupCosineSchedule,
)
from src.utils.tensors import AverageMeter, update_ema


# -----------------------------------------------------------------------------
# Device / precision helpers (CUDA | MPS | CPU)
# -----------------------------------------------------------------------------
def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp_ok(device: torch.device, amp_cfg: bool) -> bool:
    """AMP (fp16 autocast + GradScaler) only makes sense on CUDA.
    MPS has bf16 autocast in newer PyTorch but GradScaler isn't useful there.
    """
    return bool(amp_cfg) and device.type == "cuda"


class _NoOpScaler:
    """Stand-in for ``torch.cuda.amp.GradScaler`` on CPU / MPS."""
    def scale(self, loss):   return loss
    def unscale_(self, opt): pass
    def step(self, opt):     opt.step()
    def update(self):        pass


def _make_scaler(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.GradScaler(enabled=True)
    return _NoOpScaler()


def _autocast_ctx(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast(enabled=True)

    class _NullCtx:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _NullCtx()


def _target_features(
    target_encoder: torch.nn.Module,
    images: torch.Tensor,
    masks_pred: list[torch.Tensor],
) -> torch.Tensor:
    """Compute target features for every predictor block.

    1.  Run the target encoder on the full image.
    2.  Gather patch embeddings at target-mask positions.
    3.  L2-normalise (same as I-JEPA — prevents scale runaway in Euclidean
        targets; we do it before feature clipping in the hyperbolic loss).
    """
    with torch.no_grad():
        h = target_encoder(images)                       # (B, N, D)
        h = F.layer_norm(h, (h.size(-1),))
        gathered = apply_masks(h, masks_pred)            # (B*n, N_tgt, D)
    return gathered


def train_one_epoch(
    epoch: int,
    cfg: dict,
    loader,
    context_encoder,
    target_encoder,
    predictor,
    optimizer,
    lr_sched,
    wd_sched,
    mom_sched,
    loss_fn: torch.nn.Module,
    scaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    context_encoder.train()
    predictor.train()
    target_encoder.eval()

    loss_meter = AverageMeter()
    tic = time.time()
    non_blocking = device.type == "cuda"          # only CUDA benefits from this

    for it, (images, masks_enc, masks_pred) in enumerate(loader):
        images = images.to(device, non_blocking=non_blocking)
        masks_enc = [m.to(device, non_blocking=non_blocking) for m in masks_enc]
        masks_pred = [m.to(device, non_blocking=non_blocking) for m in masks_pred]

        lr = lr_sched.step()
        wd = wd_sched.step()
        m = mom_sched.step()

        with _autocast_ctx(use_amp):
            # targets (no grad)
            target_repr = _target_features(target_encoder, images, masks_pred)

            # context
            ctx_repr = context_encoder(images, masks=masks_enc)       # (B, N_ctx, D)

            # predictor
            pred_repr = predictor(ctx_repr, masks_enc, masks_pred)    # (B*n, N_tgt, D)

            loss = loss_fn(pred_repr, target_repr)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg["grad_clip"] is not None:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(context_encoder.parameters()) + list(predictor.parameters()),
                cfg["grad_clip"],
            )
        scaler.step(optimizer)
        scaler.update()

        update_ema(context_encoder, target_encoder, momentum=m)

        loss_meter.update(loss.item(), n=images.size(0))

        if it % cfg["log_every"] == 0:
            print(
                f"[epoch {epoch:3d} it {it:4d}/{len(loader)}] "
                f"loss={loss_meter.avg:.4f}  lr={lr:.2e}  wd={wd:.2e}  ema={m:.4f}  "
                f"elapsed={time.time()-tic:.1f}s",
                flush=True,
            )

    return loss_meter.avg


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = _pick_device()
    use_amp = _amp_ok(device, cfg.get("amp", False))
    print(f"Using device: {device}  |  AMP: {use_amp}")

    # Dataloader + masking
    collator = MultiBlockMaskCollator(
        input_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        enc_mask_scale=tuple(cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(cfg["pred_mask_scale"]),
        aspect_ratio=tuple(cfg["aspect_ratio"]),
        num_enc_masks=cfg["num_enc_masks"],
        num_pred_masks=cfg["num_pred_masks"],
        min_keep=cfg["min_keep"],
        allow_overlap=cfg["allow_overlap"],
    )
    loader, _ = build_pretrain_loader(
        root=cfg["data_root"],
        dataset=cfg["dataset"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        img_size=cfg["img_size"],
        collate_fn=collator,
    )

    # Models
    context_encoder, target_encoder, predictor = init_models(
        encoder_name=cfg["encoder"],
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        predictor_embed_dim=cfg["predictor_embed_dim"],
        predictor_depth=cfg["predictor_depth"],
        predictor_num_heads=cfg["predictor_num_heads"],
        device=device,
        curvature=float(cfg.get("curvature", 1.0)),
    )

    optimizer = init_optimizer(
        context_encoder, predictor,
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )

    steps_per_epoch = len(loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    lr_sched = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        start_lr=cfg["start_lr"],
        ref_lr=cfg["lr"],
        final_lr=cfg["final_lr"],
    )
    wd_sched = CosineWDSchedule(
        optimizer, ref_wd=cfg["weight_decay"],
        final_wd=cfg["final_weight_decay"], total_steps=total_steps,
    )
    mom_sched = MomentumSchedule(
        ema_start=cfg["ema_start"], ema_end=cfg["ema_end"], total_steps=total_steps,
    )

    loss_fn = build_loss(cfg)
    print(f"Loss: {loss_fn.__class__.__name__}  "
          f"(loss_type={cfg.get('loss_type', 'poincare')})")

    scaler = _make_scaler(use_amp)

    # Output
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train
    for epoch in range(cfg["epochs"]):
        avg_loss = train_one_epoch(
            epoch=epoch, cfg=cfg, loader=loader,
            context_encoder=context_encoder, target_encoder=target_encoder,
            predictor=predictor, optimizer=optimizer,
            lr_sched=lr_sched, wd_sched=wd_sched, mom_sched=mom_sched,
            loss_fn=loss_fn, scaler=scaler, device=device, use_amp=use_amp,
        )
        ckpt = {
            "epoch": epoch + 1,
            "context_encoder": context_encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "loss": avg_loss,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save(ckpt, out_dir / f"epoch{epoch+1:04d}.pt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    args = ap.parse_args()
    main(args.config)
