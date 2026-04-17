"""Custom Transformer with RoPE, optional FFN, separate attn/ffn dropout."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from jaxtyping import Float
from torch import Tensor as TT

from src.dl.dl_base import DLBase, typed


class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, T: int):
        t = torch.arange(T, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q: TT, k: TT) -> tuple[TT, TT]:
        T = q.shape[-2]
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)
        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]
        return self._apply(q, cos, sin), self._apply(k, cos, sin)

    @staticmethod
    def _apply(x: TT, cos: TT, sin: TT) -> TT:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, attn_drop: float, ffn_drop: float, rope: RoPE):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.attn_drop = attn_drop

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        if ffn_dim > 0:
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(),
                nn.Dropout(ffn_drop),
                nn.Linear(ffn_dim, d_model),
                nn.Dropout(ffn_drop),
            )
        else:
            self.norm2 = None
            self.ffn = None

    def forward(self, x: TT, attn_mask: TT | None = None) -> TT:
        h = self.norm1(x)
        q, k, v = rearrange(
            self.qkv(h), "B T (three nh hd) -> three B nh T hd",
            three=3, nh=self.n_heads,
        )
        q, k = self.rope(q, k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = rearrange(out, "B nh T hd -> B T (nh hd)")
        x = x + self.out_proj(out)

        if self.ffn is not None:
            x = x + self.ffn(self.norm2(x))
        return x


class Transformer(DLBase):
    @beartype
    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        attn_drop: float,
        ffn_drop: float,
        max_len: int = 1000,
        **kw,
    ):
        super().__init__(**kw)
        self.input_dim = input_dim * self.mimo
        self.d_model = d_model

        self.input_proj = nn.Linear(self.input_dim, d_model)
        rope = RoPE(d_model // n_heads, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, attn_drop, ffn_drop, rope)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, self.input_dim)

        self._buf: TT | None = None

    @typed
    def forward(self, x: Float[TT, "B T D"]) -> Float[TT, "B T D"]:
        h = self.input_proj(x)
        T = h.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        attn_mask = torch.zeros(T, T, device=h.device)
        attn_mask.masked_fill_(mask, float("-inf"))

        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)
        h = self.final_norm(h)
        return self.readout(h)

    @typed
    def predict_raw(self, x_t: Float[TT, "D"]) -> Float[TT, "D"]:
        x = rearrange(x_t, "D -> 1 1 D")
        if self._buf is None:
            self._buf = x
        else:
            self._buf = torch.cat([self._buf, x], dim=1)
        pred = self.forward(self._buf)
        return rearrange(pred[:, -1:], "1 1 D -> D")

    def reset_state(self) -> None:
        self._buf = None
