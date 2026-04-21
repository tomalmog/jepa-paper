"""Model / optimizer factories for Hyperbolic-JEPA training."""
from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import torch

from src.models.vision_transformer import (
    ENCODER_FACTORY,
    VisionTransformer,
    VisionTransformerPredictor,
)
from src.utils.tensors import copy_params


def _is_hyperbolic_encoder(name: str) -> bool:
    return name.startswith("hyp_")


def init_models(
    encoder_name: str,
    img_size: int,
    patch_size: int,
    predictor_embed_dim: int,
    predictor_depth: int,
    predictor_num_heads: int,
    device: torch.device,
    curvature: float = 1.0,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Build (context_encoder, target_encoder, predictor).

    If ``encoder_name`` starts with ``hyp_`` we use the fully-hyperbolic
    encoder + predictor pair (Poincaré ball with curvature ``c=curvature``);
    otherwise we use the standard Euclidean ViT pair.

    The target encoder is a deep copy of the context encoder and is updated
    by EMA; its gradients are disabled.
    """
    if encoder_name not in ENCODER_FACTORY:
        raise ValueError(f"Unknown encoder '{encoder_name}'.")

    is_hyp = _is_hyperbolic_encoder(encoder_name)
    enc_kw = {"curvature": curvature} if is_hyp else {}
    enc = ENCODER_FACTORY[encoder_name](
        img_size=img_size, patch_size=patch_size, **enc_kw,
    )
    tgt = deepcopy(enc)
    copy_params(enc, tgt)
    for p in tgt.parameters():
        p.requires_grad_(False)

    if is_hyp:
        # Import lazily so Euclidean-only configs don't pay the import cost
        # (and so the circular-import risk is contained).
        from src.models.hyperbolic_vit import HyperbolicVisionTransformerPredictor
        pred = HyperbolicVisionTransformerPredictor(
            num_patches=enc.num_patches,
            encoder_embed_dim=enc.embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            curvature=curvature,
        )
    else:
        pred = VisionTransformerPredictor(
            num_patches=enc.num_patches,
            encoder_embed_dim=enc.embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
        )
    return enc.to(device), tgt.to(device), pred.to(device)


def init_optimizer(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW with LR/WD applied only to 2-D weight tensors (same policy as I-JEPA)."""
    decay_params, no_decay_params = [], []
    for module in (encoder, predictor):
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or n.endswith(".bias") or "pos_embed" in n or "mask_token" in n:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
