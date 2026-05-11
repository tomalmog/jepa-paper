"""Fully-hyperbolic ViT building blocks on the Poincaré ball.

References
----------
- Ganea, Bécigneul, Hofmann. "Hyperbolic Neural Networks." NeurIPS 2018.
    Canonical Möbius matmul and Möbius bias used here.
- Shimizu, Mukuta, Harada. "Hyperbolic Neural Networks++." ICLR 2021.
    Tangent-space LayerNorm and activation wrappers.
- Gulcehre et al. "Hyperbolic Attention Networks." ICLR 2019.
    Tangent-space multi-head attention (what we use here).
- van Spengler, Berkhout, Mettes. "Poincaré ResNet." ICCV 2023.
    Overall recipe for hyperbolic residual blocks via Möbius addition.

Design decisions
----------------
* Activations between blocks live **on the Poincaré ball**, not in tangent.
* Residuals use **Möbius addition** (not tangent addition) so they respect
  the manifold's group structure.
* MLP linears are **Möbius matmul + Möbius bias** (Ganea et al. formula).
* Attention is **tangent-space multi-head attention**: log → standard MHA → exp.
  True hyperbolic attention (Chen et al. 2022 Lorentz model) is brittle in
  practice; the tangent-space variant is the canonical Poincaré compromise.
* Parameters stay Euclidean; standard AdamW works — no Riemannian optimizer.
* All ops are MPS-safe (no Conv2d, no AMP).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hyperbolic import (
    BALL_EPS,
    MAX_TANH_ARG,
    MIN_NORM,
    expmap0,
    logmap0,
    mobius_add,
    project,
)


# -----------------------------------------------------------------------------
# Möbius matmul (Ganea et al. 2018, Eq. 27)
# -----------------------------------------------------------------------------
def mobius_matvec(weight: torch.Tensor, x: torch.Tensor, c: float) -> torch.Tensor:
    r"""Möbius matrix-vector product on the Poincaré ball.

    .. math::
        M \otimes_c x = \frac{1}{\sqrt{c}} \tanh\!\left(
            \frac{\|Mx\|}{\|x\|} \, \mathrm{arctanh}(\sqrt{c}\,\|x\|)
        \right)\, \frac{Mx}{\|Mx\|}

    Shapes:
        weight: (out_dim, in_dim)
        x:      (..., in_dim)  on the ball
    Returns: (..., out_dim) on the ball.
    """
    sqrt_c = c ** 0.5
    x = project(x, c)

    Wx = F.linear(x, weight)                                           # Euclidean M·x
    x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    Wx_norm = Wx.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)

    at_arg = (sqrt_c * x_norm).clamp(max=1.0 - BALL_EPS)
    inner = (Wx_norm / x_norm) * torch.atanh(at_arg)
    inner = inner.clamp(max=MAX_TANH_ARG)
    scale = torch.tanh(inner) / (sqrt_c * Wx_norm)

    y = scale * Wx
    return project(y, c)


class MobiusLinear(nn.Module):
    """Fully-hyperbolic affine layer:  y = W ⊗_c x ⊕_c b.

    Weight is Euclidean, initialised like ``nn.Linear``.  Bias is a tangent
    vector at the origin that we exp-map and Möbius-add.
    """

    def __init__(self, in_dim: int, out_dim: int, c: float, bias: bool = True):
        super().__init__()
        self.c = float(c)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = mobius_matvec(self.weight, x, self.c)
        if self.bias is not None:
            b_ball = expmap0(self.bias, self.c).expand_as(out)
            out = mobius_add(out, b_ball, self.c)
        return project(out, self.c)


# -----------------------------------------------------------------------------
# Tangent-space wrappers for things that don't have a clean Möbius analogue
# -----------------------------------------------------------------------------
def hyp_layer_norm(x: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
    """LayerNorm in the tangent space at the origin, back to the ball."""
    u = logmap0(x, c)
    u = F.layer_norm(u, (u.size(-1),), eps=eps)
    return expmap0(u, c)


def hyp_activation(x: torch.Tensor, act: nn.Module, c: float) -> torch.Tensor:
    """Pointwise activation applied in the tangent space."""
    u = logmap0(x, c)
    u = act(u)
    return expmap0(u, c)


# -----------------------------------------------------------------------------
# Hyperbolic MLP (Möbius linear + tangent-wrapped GELU + dropout in tangent)
# -----------------------------------------------------------------------------
class HyperbolicMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        c: float,
        out_dim: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        self.c = float(c)
        out_dim = out_dim or in_dim
        self.fc1 = MobiusLinear(in_dim, hidden_dim, c)
        self.act = nn.GELU()
        self.fc2 = MobiusLinear(hidden_dim, out_dim, c)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        # Fused activation + dropout in tangent (avoids redundant exp/log)
        u = logmap0(x, self.c)
        u = self.drop(self.act(u))
        x = expmap0(u, self.c)
        x = self.fc2(x)
        # Fused dropout in tangent
        u = logmap0(x, self.c)
        u = self.drop(u)
        return expmap0(u, self.c)


# -----------------------------------------------------------------------------
# Tangent-space multi-head attention
# -----------------------------------------------------------------------------
class HyperbolicAttention(nn.Module):
    """Multi-head attention on ball features, done in tangent space.

    log(ball) → Euclidean MHA → exp(tangent) → ball.

    The Euclidean QKV/proj are intentional: the softmax/weighted-sum in
    hyperbolic geometry has no canonical closed form, so the accepted recipe
    is to do standard attention in the tangent space at the origin.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        c: float,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.c = float(c)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        u = logmap0(x, self.c)                                 # (B, N, D) tangent

        qkv = self.qkv(u)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q = qkv[0].contiguous(); k = qkv[1].contiguous(); v = qkv[2].contiguous()

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, D)
        out = self.proj_drop(self.proj(out))
        return expmap0(out, self.c)


# -----------------------------------------------------------------------------
# Hyperbolic block: pre-norm + tangent attention + Möbius residual,
#                   pre-norm + Möbius MLP + Möbius residual.
# -----------------------------------------------------------------------------
class HyperbolicBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        c: float,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.c = float(c)
        self.attn = HyperbolicAttention(
            dim, num_heads, c, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.mlp = HyperbolicMLP(dim, int(dim * mlp_ratio), c, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(hyp_layer_norm(x, self.c))
        x = project(mobius_add(x, y, self.c), self.c)

        y = self.mlp(hyp_layer_norm(x, self.c))
        x = project(mobius_add(x, y, self.c), self.c)
        return x
