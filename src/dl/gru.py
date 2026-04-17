import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from haste_pytorch import GRU as HasteGRU
from jaxtyping import Float
from scipy.stats import special_ortho_group
from torch import Tensor as TT

from src.dl.dl_base import DLBase, typed
from src.dl.dropout import LockedDropout, WeightDrop


def _ortho_init_param(param: TT):
    """Initialize a weight_hh-shaped parameter with orthogonal matrices (one per gate)."""
    h = param.shape[1]
    n_gates = param.shape[0] // h
    for g in range(n_gates):
        ortho = special_ortho_group.rvs(h).astype(np.float32)
        param.data[g * h : (g + 1) * h] = TT(ortho)


def _ortho_init(gru: nn.GRU):
    for name, param in gru.named_parameters():
        if "weight_hh" not in name:
            continue
        _ortho_init_param(param)


class ZoneoutGRU(nn.Module):
    """GRU with per-step zoneout on hidden state. Uses GRUCell loop."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 weight_drop: float, layer_drop: float, zoneout: float, ortho_init: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.weight_drop = weight_drop
        self.layer_drop = layer_drop
        self.zoneout = zoneout

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell = nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            if ortho_init:
                _ortho_init_param(cell.weight_hh)
            self.cells.append(cell)

    def _loop(self, x: TT, h: list[TT], zo_masks: TT | None) -> tuple[TT, list[TT]]:
        T = x.shape[1]
        outputs = []
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                new_h = cell(inp, h[i])

                if zo_masks is not None:
                    m = zo_masks[i, t]
                    new_h = m * h[i] + (1 - m) * new_h

                h[i] = new_h
                if i < self.num_layers - 1 and self.layer_drop > 0:
                    inp = F.dropout(new_h, self.layer_drop, self.training)
                else:
                    inp = new_h
            outputs.append(h[-1])

        return torch.stack(outputs, dim=1), h

    def forward(self, x: TT, h0: TT | None = None) -> tuple[TT, TT]:
        B, T, _ = x.shape
        device = x.device

        if h0 is None:
            h = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
        else:
            h = [h0[i] for i in range(self.num_layers)]

        # apply weight dropout once per forward (same mask for all timesteps)
        originals = {}
        if self.training and self.weight_drop > 0:
            for i, cell in enumerate(self.cells):
                originals[i] = cell._parameters["weight_hh"]
                cell._parameters["weight_hh"] = F.dropout(originals[i], p=self.weight_drop, training=True)

        # precompute zoneout masks
        if self.zoneout > 0:
            if self.training:
                zo_masks = torch.bernoulli(x.new_full((self.num_layers, T, B, self.hidden_size), self.zoneout))
            else:
                zo_masks = x.new_full((self.num_layers, T, 1, 1), self.zoneout)
        else:
            zo_masks = None

        out, h = self._loop(x, h, zo_masks)

        # restore original weights
        for i, orig in originals.items():
            self.cells[i]._parameters["weight_hh"] = orig

        h_n = torch.stack(h, dim=0)
        return out, h_n


class StackedHasteGRU(nn.Module):
    """Multi-layer wrapper around haste_pytorch.GRU (which is single-layer)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 weight_drop: float, layer_drop: float, zoneout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_drop = layer_drop

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(HasteGRU(
                input_size if i == 0 else hidden_size,
                hidden_size,
                batch_first=True,
                dropout=weight_drop,
                zoneout=zoneout,
            ))
        self.lockdrop = LockedDropout()

    def forward(self, x: TT, h0: TT | None = None) -> tuple[TT, TT]:
        h_out = []
        inp = x
        for i, layer in enumerate(self.layers):
            hi = h0[i:i+1] if h0 is not None else None
            inp, h_n = layer(inp, hi)
            h_out.append(h_n)
            if i < self.num_layers - 1 and self.layer_drop > 0:
                inp = self.lockdrop(inp, self.layer_drop)
        return inp, torch.cat(h_out, dim=0)


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
        layer_norm: bool = False,
        learned_h0: bool = False,
        zoneout: float = 0.0,
        use_haste: bool = False,
        **kw,
    ):
        super().__init__(**kw)
        self.input_dim = input_dim * self.mimo
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_drop = input_drop
        self.output_drop = output_drop

        if use_haste:
            self.gru = StackedHasteGRU(
                self.input_dim, hidden_dim, num_layers,
                weight_drop=weight_drop, layer_drop=layer_drop,
                zoneout=zoneout,
            )
        elif zoneout > 0:
            self.gru = ZoneoutGRU(
                self.input_dim, hidden_dim, num_layers,
                weight_drop=weight_drop, layer_drop=layer_drop,
                zoneout=zoneout, ortho_init=ortho_init,
            )
        else:
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
        self.ln = nn.LayerNorm(hidden_dim) if layer_norm else None
        self.readout = nn.Linear(hidden_dim, self.input_dim)
        self._init_aux_heads(hidden_dim, self.input_dim)

        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim)) if learned_h0 else None
        self._h: TT | None = None

    def _init_hidden(self, batch_size: int) -> TT | None:
        if self.h0 is None:
            return None
        return self.h0.expand(-1, batch_size, -1).contiguous()

    def forward_hidden(self, x):
        x = self.lockdrop(x, self.input_drop)
        h0 = self._init_hidden(x.shape[0])
        out, _ = self.gru(x) if h0 is None else self.gru(x, h0)
        if self.ln is not None:
            out = self.ln(out)
        return self.lockdrop(out, self.output_drop)

    @typed
    def forward(self, x: Float[TT, "B T D"]) -> Float[TT, "B T D"]:
        return self.readout(self.forward_hidden(x))

    @typed
    def predict_raw(self, x_t: Float[TT, "D"]) -> Float[TT, "D"]:
        x = rearrange(x_t, "D -> 1 1 D")
        h = self._h if self._h is not None else self._init_hidden(1)
        out, self._h = self.gru(x, h)
        if self.ln is not None:
            out = self.ln(out)
        pred = self.readout(out)
        return rearrange(pred, "1 1 D -> D")

    def reset_state(self) -> None:
        self._h = None
