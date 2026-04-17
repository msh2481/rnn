import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from jaxtyping import Float
from scipy.stats import special_ortho_group
from torch import Tensor as TT

from src.dl.dl_base import DLBase, typed
from src.dl.dropout import LockedDropout, WeightDrop


def _ortho_init(gru: nn.GRU):
    for name, param in gru.named_parameters():
        if "weight_hh" not in name:
            continue
        h = param.shape[1]
        n_gates = param.shape[0] // h
        for g in range(n_gates):
            ortho = special_ortho_group.rvs(h).astype(np.float32)
            param.data[g * h : (g + 1) * h] = TT(ortho)


class CausalWindowAttention(nn.Module):
    def __init__(
        self, hidden_dim: int, n_heads: int, window_size: int, attn_drop: float
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.window_size = window_size
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    @typed
    def forward(self, x: Float[TT, "B T H"]) -> Float[TT, "B T H"]:
        B, T, H = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "B T (three nh hd) -> three B nh T hd", three=3, nh=self.n_heads
        )

        scores = torch.einsum("bnid,bnjd->bnij", q, k) / math.sqrt(self.head_dim)

        # causal + window mask: position i attends to j where i-W+1 <= j <= i
        idx = torch.arange(T, device=x.device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (T, T): diff[i,j] = i - j
        mask = (diff < 0) | (diff >= self.window_size)  # True = masked out
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.training and self.attn_drop > 0:
            attn = F.dropout(attn, p=self.attn_drop)

        out = torch.einsum("bnij,bnjd->bnid", attn, v)
        out = rearrange(out, "B nh T hd -> B T (nh hd)")
        return self.out_proj(out)


class GRUAttn(DLBase):
    @beartype
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_heads: int,
        window_size: int,
        weight_drop: float,
        input_drop: float,
        output_drop: float,
        attn_drop: float,
        layer_drop: float,
        ortho_init: bool,
        **kw,
    ):
        super().__init__(**kw)
        self.input_dim = input_dim * self.mimo
        self.hidden_dim = hidden_dim
        self.input_drop = input_drop
        self.output_drop = output_drop
        self.window_size = window_size

        raw_gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=layer_drop if num_layers > 1 else 0.0,
        )
        if ortho_init:
            _ortho_init(raw_gru)
        self.gru = WeightDrop(raw_gru, weight_drop)
        self.lockdrop = LockedDropout()
        self.attn = CausalWindowAttention(hidden_dim, n_heads, window_size, attn_drop)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, self.input_dim)

        self._h: TT | None = None
        self._buf: TT | None = None  # buffer of past GRU outputs for inference

    @typed
    def forward(self, x: Float[TT, "B T D"]) -> Float[TT, "B T D"]:
        x = self.lockdrop(x, self.input_drop)
        gru_out, _ = self.gru(x)
        gru_out = self.lockdrop(gru_out, self.output_drop)
        attn_out = self.attn(gru_out)
        out = self.attn_norm(gru_out + attn_out)
        return self.readout(out)

    @typed
    def predict_raw(self, x_t: Float[TT, "D"]) -> Float[TT, "D"]:
        x = rearrange(x_t, "D -> 1 1 D")
        gru_out, self._h = self.gru(x, self._h)  # (1, 1, H)

        # append to buffer, trim to window
        if self._buf is None:
            self._buf = gru_out
        else:
            self._buf = torch.cat([self._buf, gru_out], dim=1)
            if self._buf.shape[1] > self.window_size:
                self._buf = self._buf[:, -self.window_size :]

        # attend over buffer
        attn_out = self.attn(self._buf)
        last = attn_out[:, -1:]  # (1, 1, H)
        gru_last = self._buf[:, -1:]
        out = self.attn_norm(gru_last + last)
        pred = self.readout(out)
        return rearrange(pred, "1 1 D -> D")

    def reset_state(self) -> None:
        self._h = None
        self._buf = None
