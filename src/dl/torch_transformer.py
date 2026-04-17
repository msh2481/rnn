"""Baseline Transformer using nn.TransformerDecoder.

Causal window mask generated externally. No RoPE, no KV cache (brute-force
recompute for inference). Learned positional embeddings.
"""

import math

import torch
import torch.nn as nn
from beartype import beartype
from einops import rearrange
from jaxtyping import Float
from torch import Tensor as TT

from src.dl.dl_base import DLBase, typed


class TorchTransformer(DLBase):
    @beartype
    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        max_len: int = 1000,
        **kw,
    ):
        super().__init__(**kw)
        self.input_dim = input_dim * self.mimo
        self.d_model = d_model
        self.max_len = max_len

        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.readout = nn.Linear(d_model, self.input_dim)

        # inference buffer
        self._buf: TT | None = None
        self._step: int = 0

    def _causal_mask(self, T: int, device: torch.device) -> TT:
        """Causal attention mask. True = blocked."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    @typed
    def forward(self, x: Float[TT, "B T D"]) -> Float[TT, "B T D"]:
        B, T, _ = x.shape
        h = self.input_proj(x)
        positions = torch.arange(T, device=x.device)
        h = h + self.pos_emb(positions)

        mask = self._causal_mask(T, x.device)
        h = self.decoder(h, h, tgt_mask=mask, memory_mask=mask)
        return self.readout(h)

    @typed
    def predict_raw(self, x_t: Float[TT, "D"]) -> Float[TT, "D"]:
        x = rearrange(x_t, "D -> 1 1 D")

        if self._buf is None:
            self._buf = x
        else:
            self._buf = torch.cat([self._buf, x], dim=1)

        # trim to max_len
        if self._buf.shape[1] > self.max_len:
            self._buf = self._buf[:, -self.max_len:]
            self._step = self.max_len - 1
        else:
            self._step = self._buf.shape[1] - 1

        # full forward on buffer (brute-force, no KV cache)
        pred = self.forward(self._buf)
        return rearrange(pred[:, -1:], "1 1 D -> D")

    def reset_state(self) -> None:
        self._buf = None
        self._step = 0
