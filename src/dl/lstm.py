import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from jaxtyping import Float
from torch import Tensor as TT

from src.dl.dl_base import DLBase, typed


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
        for name in self._weight_names:
            raw = getattr(rnn, name)
            delattr(rnn, name)
            self.register_parameter(f"raw_{name}", raw)

    def _patch_weights(self):
        for name in self._weight_names:
            raw = getattr(self, f"raw_{name}")
            if self.training:
                dropped = F.dropout(raw, p=self.weight_drop, training=True)
            else:
                dropped = raw
            # bypass nn.Module.__setattr__ which rejects non-Parameter tensors
            self.rnn._parameters[name] = dropped

    def forward(self, x, hx=None):
        self._patch_weights()
        return self.rnn(x, hx)


class LSTM(DLBase):
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
        **kw,
    ):
        super().__init__(**kw)
        self.input_dim = input_dim * self.mimo
        self.hidden_dim = hidden_dim
        self.input_drop = input_drop
        self.output_drop = output_drop

        raw_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.lstm = WeightDrop(raw_lstm, weight_drop)
        self.lockdrop = LockedDropout()
        self.readout = nn.Linear(hidden_dim, self.input_dim)

        self._h: tuple[TT, TT] | None = None

    @typed
    def forward(self, x: Float[TT, "B T D"]) -> Float[TT, "B T D"]:
        x = self.lockdrop(x, self.input_drop)
        out, _ = self.lstm(x)
        out = self.lockdrop(out, self.output_drop)
        return self.readout(out)

    @typed
    def predict_raw(self, x_t: Float[TT, "D"]) -> Float[TT, "D"]:
        x = rearrange(x_t, "D -> 1 1 D")
        out, self._h = self.lstm(x, self._h)
        pred = self.readout(out)
        return rearrange(pred, "1 1 D -> D")

    def reset_state(self) -> None:
        self._h = None


if __name__ == "__main__":
    from functools import partial

    import torch.optim as optim

    from src.utils import train_and_eval

    model = LSTM(
        name="lstm_baseline",
        input_dim=32,
        hidden_dim=64,
        num_layers=1,
        weight_drop=0.2,
        input_drop=0.1,
        output_drop=0.1,
        n_epochs=20,
        batch_size=16,
        mimo=1,
        optimizer_fn=partial(optim.Adam, lr=1e-3),
        scheduler_fn=None,
    )
    model.fit()
    train_and_eval(model)
