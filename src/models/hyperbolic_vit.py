"""Fully-hyperbolic Vision Transformer for Hyperbolic-JEPA.

Structure mirrors ``src.models.vision_transformer`` exactly (same depth,
heads, masking interface, predictor shape) so the three models we compare
differ only in the intended places.

Encoder data flow::

    images
      │  (Euclidean Linear patch embed — same as Euclidean ViT)
      ▼
    tangent features  +  2-D sincos pos_embed
      │  (add in tangent, then map onto the ball — this is the one place
      │   where we leave Euclidean land)
      ▼
    expmap₀_c   →  Poincaré ball
      │
      ▼   HyperbolicBlock × depth  (Möbius linears, Möbius residuals,
      │                             tangent-wrapped attention, pre-norm in tangent)
      ▼
    hyp_layer_norm (tangent LN + back to ball)
      │
      ▼
    logmap₀_c   →  tangent vector
      │
      ▼
    output  (same tangent-vector interface as the Euclidean ViT so the
             shared loss / eval pipeline can consume either model)

What's hyperbolic vs. tangent-wrapped
-------------------------------------
* **Genuinely Möbius-native:** residual connections, MLP linear layers
  (fc1/fc2), bias addition.
* **Tangent-wrapped (log → Euclidean → exp):** attention QKV+softmax+proj
  (no canonical closed form for hyperbolic softmax), pointwise GELU,
  LayerNorm, patch embed, predictor's dim-change projections (``predictor_embed``
  and ``predictor_proj``).

The predictor's *block stack* is fully hyperbolic; only its thin I/O
projections are Euclidean Linear, which is the conventional choice in the
Poincaré-ball literature for dim-change affinities.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.masks.multiblock import apply_masks
from src.models.hyperbolic import expmap0, logmap0
from src.models.hyperbolic_layers import HyperbolicBlock, hyp_layer_norm
from src.models.vision_transformer import (
    PatchEmbed,
    _gather_pos,
    _init_weights,
    get_2d_sincos_pos_embed,
)


# -----------------------------------------------------------------------------
# Fully-hyperbolic context / target encoder
# -----------------------------------------------------------------------------
class HyperbolicVisionTransformer(nn.Module):
    """ViT whose block stack operates on the Poincaré ball.

    Same masking interface as ``VisionTransformer``: ``masks=None`` returns
    features for every patch; passing mask indices keeps only those patches
    (used by I-JEPA to feed the *context* encoder).
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.c = float(curvature)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList([
            HyperbolicBlock(
                dim=embed_dim, num_heads=num_heads, c=self.c,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, masks: Optional[list] = None) -> torch.Tensor:
        # 1) Euclidean patch embedding + Euclidean sincos pos-embed.
        x = self.patch_embed(x) + self.pos_embed                  # (B, N, D)  tangent
        if masks is not None:
            x = apply_masks(x, masks)                             # drop masked patches

        # 2) Enter the Poincaré ball.
        x = expmap0(x, self.c)

        # 3) Hyperbolic block stack.
        for blk in self.blocks:
            x = blk(x)

        # 4) Final normalisation in tangent, then return a tangent vector so
        #    the downstream loss / eval code can stay identical to the Euclidean
        #    case. (Internally all the ball dynamics still happened; this is
        #    just returning the representation in its tangent-at-origin chart.)
        x = hyp_layer_norm(x, self.c)
        x = logmap0(x, self.c)
        return x                                                   # (B, N, D) tangent


# -----------------------------------------------------------------------------
# Fully-hyperbolic predictor
# -----------------------------------------------------------------------------
class HyperbolicVisionTransformerPredictor(nn.Module):
    """I-JEPA predictor whose transformer stack is fully hyperbolic.

    Matches the shape and interface of ``VisionTransformerPredictor``. The
    thin input / output dim-change projections stay Euclidean (tangent),
    which is the common practice in Poincaré-ball literature for affine
    maps that cross dimensions.
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int,
        predictor_embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.c = float(curvature)
        self.num_patches = num_patches

        # Tangent-space projections at the predictor's I/O boundaries.
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        pos_embed = get_2d_sincos_pos_embed(
            predictor_embed_dim, int(math.sqrt(num_patches))
        )
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList([
            HyperbolicBlock(
                dim=predictor_embed_dim, num_heads=num_heads, c=self.c,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        self.predictor_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(
        self,
        ctx_tokens: torch.Tensor,
        masks_enc: list,
        masks_pred: list,
    ) -> torch.Tensor:
        assert len(masks_enc) == 1, "I-JEPA uses a single context block."

        # Dim-change projection + predictor's own pos-embed (tangent).
        x = self.predictor_embed(ctx_tokens)
        x = x + _gather_pos(self.pos_embed, masks_enc[0])                # (B, Nc, Dp)

        # Build per-block mask tokens with their positional slots (tangent).
        n_blocks = len(masks_pred)
        x = x.repeat(n_blocks, 1, 1)                                     # (B·n, Nc, Dp)
        mask_tokens = []
        for m in masks_pred:
            mt = self.mask_token.repeat(m.size(0), m.size(1), 1)
            mt = mt + _gather_pos(self.pos_embed, m)
            mask_tokens.append(mt)
        mask_tokens = torch.cat(mask_tokens, dim=0)                      # (B·n, Nt, Dp)

        # Concatenate and enter the ball for the hyperbolic block stack.
        n_ctx = x.size(1)
        x = torch.cat([x, mask_tokens], dim=1)                           # (B·n, Nc+Nt, Dp)
        x = expmap0(x, self.c)

        for blk in self.blocks:
            x = blk(x)

        # Exit the ball and return only the target slots, re-projected.
        x = hyp_layer_norm(x, self.c)
        x = logmap0(x, self.c)

        x = x[:, n_ctx:]
        return self.predictor_proj(x)


# -----------------------------------------------------------------------------
# Factories (mirror sizes of vit_tiny / vit_small so the A/B/C uses the
# same parameter count modulo Möbius layer weights — which have the same
# shape as Euclidean Linear weights).
# -----------------------------------------------------------------------------
def hyp_vit_tiny(img_size: int = 32, patch_size: int = 4,
                 curvature: float = 1.0, **kw):
    return HyperbolicVisionTransformer(
        img_size=img_size, patch_size=patch_size,
        embed_dim=192, depth=12, num_heads=3,
        curvature=curvature, **kw,
    )


def hyp_vit_small(img_size: int = 32, patch_size: int = 4,
                  curvature: float = 1.0, **kw):
    return HyperbolicVisionTransformer(
        img_size=img_size, patch_size=patch_size,
        embed_dim=384, depth=12, num_heads=6,
        curvature=curvature, **kw,
    )


HYPERBOLIC_ENCODER_FACTORY = {
    "hyp_vit_tiny": hyp_vit_tiny,
    "hyp_vit_small": hyp_vit_small,
}
