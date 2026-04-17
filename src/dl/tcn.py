import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from collections import deque
from dataclasses import dataclass
from einops import reduce
from jaxtyping import Float
from torch import Tensor as TT

from src.dl.dl_base import DLBase, typed


@dataclass
class LayerSpec:
    kernel_size: int
    channels: int
    n_rep: int = 1
    ln: bool = False
    nonlin: bool = True
    dropout_mul: float = 1.0
    separable: bool = False


class ConvLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, groups: int,
                 ln: bool, nonlin: bool, dropout: float, separable: bool = False):
        super().__init__()
        self.pad = kernel_size - 1
        self.separable = separable
        if separable:
            # depthwise (each channel independent over time) + pointwise (mixes channels within each group)
            self.dw = nn.Conv1d(channels, channels, kernel_size, groups=channels)
            self.pw = nn.Conv1d(channels, channels, 1, groups=groups)
        else:
            self.conv = nn.Conv1d(channels, channels, kernel_size, groups=groups)
        self.norm = nn.LayerNorm(channels) if ln else None
        self.nonlin = nonlin
        self.dropout = dropout

    def forward(self, x: TT) -> TT:
        res = x
        h = x
        if self.norm is not None:
            h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = F.dropout(h, self.dropout, self.training)
        h = F.pad(h, (self.pad, 0))
        if self.separable:
            h = self.dw(h)
            h = self.pw(h)
        else:
            h = self.conv(h)
        if self.nonlin:
            h = F.gelu(h)
        return h + res


class GroupedReadout(nn.Module):
    def __init__(self, in_channels: int, out_features: int, groups: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_features * groups, 1, groups=groups)
        self.groups = groups

    def forward(self, x: TT) -> TT:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if self.groups > 1:
            h = reduce(h, "B T (G D) -> B T D", "mean", G=self.groups)
        return h


class TCN(DLBase):
    @beartype
    def __init__(
        self,
        *,
        input_dim: int,
        layers: list[LayerSpec],
        groups: int = 1,
        dropout: float = 0.1,
        **kw,
    ):
        super().__init__(**kw)
        self.input_dim = input_dim * self.mimo
        self.groups = groups

        chs = [s.channels for s in layers]
        first_ch, last_ch = chs[0], chs[-1]

        self.proj_in = nn.Conv1d(self.input_dim, first_ch, 1)

        self.convs = nn.ModuleList()
        self.projs = nn.ModuleDict()
        self._n_reps = []

        prev_ch = first_ch
        for i, spec in enumerate(layers):
            ch = spec.channels
            if prev_ch != ch:
                self.projs[str(i)] = nn.Conv1d(prev_ch, ch, 1)
            self.convs.append(ConvLayer(
                ch, spec.kernel_size, groups,
                spec.ln, spec.nonlin,
                dropout * spec.dropout_mul,
                separable=spec.separable,
            ))
            self._n_reps.append(spec.n_rep)
            prev_ch = ch

        self.readout = GroupedReadout(last_ch, self.input_dim, groups)
        self._init_aux_heads(last_ch, self.input_dim)

        self._receptive_field = 1 + sum(s.n_rep * (s.kernel_size - 1) for s in layers)
        self._buf: deque | None = None

    def forward_hidden(self, x):
        h = x.transpose(1, 2)
        h = self.proj_in(h)
        for i, (conv, n_rep) in enumerate(zip(self.convs, self._n_reps)):
            if str(i) in self.projs:
                h = self.projs[str(i)](h)
            for _ in range(n_rep):
                h = conv(h)
        return h.transpose(1, 2)

    @typed
    def forward(self, x: Float[TT, "B T D"]) -> Float[TT, "B T D"]:
        return self.readout(self.forward_hidden(x))

    @typed
    def predict_raw(self, x_t: Float[TT, "D"]) -> Float[TT, "D"]:
        if self._buf is None:
            self._buf = deque(maxlen=self._receptive_field)
        self._buf.append(x_t)
        buf_list = list(self._buf)
        while len(buf_list) < self._receptive_field:
            buf_list.insert(0, torch.zeros_like(x_t))
        x = torch.stack(buf_list).unsqueeze(0)
        out = self.forward(x)
        return out[0, -1]

    def reset_state(self) -> None:
        self._buf = None
