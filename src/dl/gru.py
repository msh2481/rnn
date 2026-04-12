import numpy as np
import torch.nn as nn
from beartype import beartype
from einops import rearrange
from jaxtyping import Float
from scipy.stats import special_ortho_group
from torch import Tensor as TT

from src.dl.dl_base import DLBase, typed
from src.dl.dropout import LockedDropout, WeightDrop


def _ortho_init(gru: nn.GRU):
    """Initialize weight_hh with orthogonal matrices (one per gate)."""
    for name, param in gru.named_parameters():
        if "weight_hh" not in name:
            continue
        h = param.shape[1]
        n_gates = param.shape[0] // h
        for g in range(n_gates):
            ortho = special_ortho_group.rvs(h).astype(np.float32)
            param.data[g * h : (g + 1) * h] = TT(ortho)


class GRU(DLBase):
    @beartype
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        weight_drop: float,
        input_drop: float,
        output_drop: float,
        layer_drop: float,
        ortho_init: bool,
        **kw,
    ):
        super().__init__(**kw)
        self.input_dim = input_dim * self.mimo
        self.hidden_dim = hidden_dim
        self.input_drop = input_drop
        self.output_drop = output_drop

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
        self.readout = nn.Linear(hidden_dim, self.input_dim)

        self._h: TT | None = None

    @typed
    def forward(self, x: Float[TT, "B T D"]) -> Float[TT, "B T D"]:
        x = self.lockdrop(x, self.input_drop)
        out, _ = self.gru(x)
        out = self.lockdrop(out, self.output_drop)
        return self.readout(out)

    @typed
    def predict_raw(self, x_t: Float[TT, "D"]) -> Float[TT, "D"]:
        x = rearrange(x_t, "D -> 1 1 D")
        out, self._h = self.gru(x, self._h)
        pred = self.readout(out)
        return rearrange(pred, "1 1 D -> D")

    def reset_state(self) -> None:
        self._h = None
