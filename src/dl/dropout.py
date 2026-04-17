import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor as TT

from src.dl.dl_base import typed


class LockedDropout(nn.Module):
    @typed
    def forward(self, x: Float[TT, "B T D"], dropout: float) -> Float[TT, "B T D"]:
        if not self.training or dropout == 0:
            return x
        mask = x.new_empty(x.shape[0], 1, x.shape[2]).bernoulli_(1 - dropout) / (1 - dropout)
        return x * mask


class WeightDrop(nn.Module):
    def __init__(self, rnn: nn.RNNBase, weight_drop: float):
        super().__init__()
        self.rnn = rnn
        self.weight_drop = weight_drop
        self._weight_names = [
            name for name in dir(rnn) if name.startswith("weight_hh_l")
        ]
        assert self._weight_names, f"No weight_hh_l* found in {type(rnn).__name__}"

    def forward(self, x, hx=None):
        if self.training and self.weight_drop > 0:
            originals = {}
            for name in self._weight_names:
                originals[name] = self.rnn._parameters[name]
                self.rnn._parameters[name] = F.dropout(originals[name], p=self.weight_drop, training=True)

        result = self.rnn(x) if hx is None else self.rnn(x, hx)

        if self.training and self.weight_drop > 0:
            for name in self._weight_names:
                self.rnn._parameters[name] = originals[name]

        return result
