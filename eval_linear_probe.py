"""
Linear-probe evaluation of a pretrained Hyperbolic-JEPA encoder.

Freezes the target encoder, extracts mean-pooled patch features
(in Euclidean space), and trains a single linear classifier on
CIFAR-10 / CIFAR-100 train → evaluates on test.

Optionally applies the log-map first so the probe sees Euclidean
"tangent" coordinates around the ball origin — this is the right
thing to do if your representations have drifted toward the ball
boundary during pretraining.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.datasets.cifar import build_eval_loaders
from src.helper import init_models
from src.models.hyperbolic import clip_feature, logmap0


@torch.no_grad()
def extract_features(encoder, loader, device, curvature=None, clip_radius=None):
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        h = encoder(x)                                   # (B, N, D)
        h = F.layer_norm(h, (h.size(-1),))
        h = h.mean(dim=1)                                # global pool
        if curvature is not None:
            h = clip_feature(h, clip_radius)
            h = logmap0(
                # exp_0 then log_0 is a no-op mathematically, so apply
                # log_0 directly on the (stable) clipped features.
                h, curvature,
            )
        feats.append(h.cpu())
        labels.append(y)
    return torch.cat(feats, 0), torch.cat(labels, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", "-ckpt", required=True)
    ap.add_argument("--use-tangent", action="store_true",
                    help="Apply log_0 to features before the linear probe.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    context_encoder, target_encoder, _ = init_models(
        encoder_name=cfg["encoder"],
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        predictor_embed_dim=cfg["predictor_embed_dim"],
        predictor_depth=cfg["predictor_depth"],
        predictor_num_heads=cfg["predictor_num_heads"],
        device=device,
        curvature=float(cfg.get("curvature", 1.0)),
    )
    target_encoder.load_state_dict(ckpt["target_encoder"])

    train_loader, test_loader, n_classes = build_eval_loaders(
        root=cfg["data_root"],
        dataset=cfg["dataset"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=cfg["img_size"],
    )

    curv = cfg["curvature"] if args.use_tangent else None
    clip = cfg["clip_radius"] if args.use_tangent else None

    Xtr, ytr = extract_features(target_encoder, train_loader, device,
                                curvature=curv, clip_radius=clip)
    Xte, yte = extract_features(target_encoder, test_loader, device,
                                curvature=curv, clip_radius=clip)
    print(f"train feats: {Xtr.shape}   test feats: {Xte.shape}")

    # L2-normalise features (standard linear-probe recipe)
    Xtr = F.normalize(Xtr, dim=1)
    Xte = F.normalize(Xte, dim=1)

    probe = nn.Linear(Xtr.size(1), n_classes).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xte, yte = Xte.to(device), yte.to(device)

    for ep in range(args.epochs):
        probe.train()
        perm = torch.randperm(Xtr.size(0), device=device)
        for i in range(0, Xtr.size(0), args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = probe(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        probe.eval()
        with torch.no_grad():
            acc = (probe(Xte).argmax(1) == yte).float().mean().item()
        print(f"[probe ep {ep:3d}] loss={loss.item():.4f}  test_acc={acc*100:.2f}")


if __name__ == "__main__":
    main()
