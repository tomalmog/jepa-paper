"""
Vision Transformer for Hyperbolic-JEPA.

Minimal faithful port of the ViT used in facebookresearch/ijepa, with
the same patch-embed / block / attention structure so weights transfer
cleanly.  No CLS token (I-JEPA uses only patch tokens).
"""
from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masks.multiblock import apply_masks


# -----------------------------------------------------------------------------
# Positional embedding helper (2-D sincos, same as MAE / I-JEPA)
# -----------------------------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)            # (2, H, W)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)                              # (H*W, D)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


# -----------------------------------------------------------------------------
# Core modules
# -----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Non-overlapping patch embedding.

    Historically this is implemented as ``nn.Conv2d(in_chans, dim,
    kernel_size=patch_size, stride=patch_size)``.  That form is
    mathematically *identical* to unfolding the image into flat patches
    and applying a single ``nn.Linear``, but on PyTorch 2.5.1 + MPS the
    strided-conv backward produces non-contiguous gradients that blow
    up the first downstream transpose/attention ("view size is not
    compatible").  We therefore use the Linear form, which works on
    CPU, CUDA and MPS alike and compiles to the same GEMM on all three.
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        # (B, C, H, W) -> (B, C, gH, p, gW, p) -> (B, gH, gW, C, p, p)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # -> (B, N, C*p*p) where N = gH * gW
        x = x.view(B, self.num_patches, C * p * p)
        return self.proj(x)                                   # (B, N, D)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                 out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: aggressive .contiguous() calls are *required* on MPS.
        # Without them, autograd hits
        #   "view size is not compatible with input tensor's size and stride"
        # at random points in the graph.  They are effectively free on CUDA.
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()                # (3, B, H, N, d)
        q = qkv[0].contiguous()
        k = qkv[1].contiguous()
        v = qkv[2].contiguous()

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# Context / Target encoder (a plain ViT)
# -----------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    """ViT that optionally drops masked patches before processing.

    - When ``masks`` is None it returns features for every patch.
    - When ``masks`` is a list of index tensors it keeps only those patches
      (this is how I-JEPA feeds the *context* encoder).
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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop_rate, attn_drop_rate, norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, masks: Optional[list] = None) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed                      # (B, N, D)
        if masks is not None:
            x = apply_masks(x, masks)                                 # keep unmasked
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# -----------------------------------------------------------------------------
# Predictor (smaller ViT that maps context tokens -> target tokens)
# -----------------------------------------------------------------------------
class VisionTransformerPredictor(nn.Module):
    """I-JEPA predictor.

    Inputs  : context tokens from the context encoder.
    Outputs : predicted embeddings at the positions of the target masks.

    Implementation mirrors ``src.models.vision_transformer_predictor`` in
    facebookresearch/ijepa: the predictor has its own (learned) positional
    embedding, learnable mask tokens, and projects to / from its narrower
    width.
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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_patches = num_patches
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        pos_embed = get_2d_sincos_pos_embed(
            predictor_embed_dim, int(math.sqrt(num_patches))
        )
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList([
            Block(predictor_embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop_rate, attn_drop_rate, norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, ctx_tokens: torch.Tensor,
                masks_enc: list, masks_pred: list) -> torch.Tensor:
        """
        Parameters
        ----------
        ctx_tokens : (B, N_ctx, D_enc)   context encoder output
        masks_enc  : list of (B, N_ctx)  indices of context patches
        masks_pred : list of (B, N_tgt)  indices of target patches (per block)

        Returns
        -------
        (B * n_blocks, N_tgt, D_enc)  predicted target embeddings
        """
        assert len(masks_enc) == 1, "I-JEPA uses a single context block."

        # Project context to predictor dim and add its positional embedding
        x = self.predictor_embed(ctx_tokens)
        x_pos = _gather_pos(self.pos_embed, masks_enc[0])
        x = x + x_pos                                                 # (B, N_ctx, D_pred)

        # Build mask tokens for each target block (positional info only)
        n_blocks = len(masks_pred)
        # Repeat context along the block dim
        x = x.repeat(n_blocks, 1, 1)                                  # (B*n, N_ctx, D)
        mask_tokens = []
        for m in masks_pred:
            mt = self.mask_token.repeat(m.size(0), m.size(1), 1)      # (B, N_tgt, D)
            mt = mt + _gather_pos(self.pos_embed, m)
            mask_tokens.append(mt)
        mask_tokens = torch.cat(mask_tokens, dim=0)                   # (B*n, N_tgt, D)

        # Concat and run the predictor transformer
        n_ctx = x.size(1)
        x = torch.cat([x, mask_tokens], dim=1)                        # (B*n, N_ctx+N_tgt, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Return only the predicted target-slot embeddings, re-projected
        x = x[:, n_ctx:]
        return self.predictor_proj(x)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def _gather_pos(pos_embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather positional embeddings at indices ``mask``.

    pos_embed: (1, N_total, D)
    mask:      (B, N_sub)   -- long indices into N_total
    returns:   (B, N_sub, D)
    """
    B = mask.size(0)
    pe = pos_embed.expand(B, -1, -1)                                  # (B, N, D)
    idx = mask.unsqueeze(-1).expand(-1, -1, pe.size(-1))
    return pe.gather(1, idx)


# -----------------------------------------------------------------------------
# Factory helpers (tiny / small configs that run on a single GPU)
# -----------------------------------------------------------------------------
def vit_tiny(img_size: int = 32, patch_size: int = 4, **kw):
    return VisionTransformer(img_size=img_size, patch_size=patch_size,
                             embed_dim=192, depth=12, num_heads=3, **kw)


def vit_small(img_size: int = 32, patch_size: int = 4, **kw):
    return VisionTransformer(img_size=img_size, patch_size=patch_size,
                             embed_dim=384, depth=12, num_heads=6, **kw)


def vit_base(img_size: int = 32, patch_size: int = 4, **kw):
    return VisionTransformer(img_size=img_size, patch_size=patch_size,
                             embed_dim=768, depth=12, num_heads=12, **kw)


ENCODER_FACTORY = {"vit_tiny": vit_tiny, "vit_small": vit_small, "vit_base": vit_base}


def _register_hyperbolic_encoders() -> None:
    """Merge the hyperbolic encoder factories into ENCODER_FACTORY at import.

    Deferred so that ``hyperbolic_vit`` (which itself imports from this
    module) doesn't trigger a circular import during module initialisation.
    """
    from src.models.hyperbolic_vit import HYPERBOLIC_ENCODER_FACTORY
    ENCODER_FACTORY.update(HYPERBOLIC_ENCODER_FACTORY)


_register_hyperbolic_encoders()
