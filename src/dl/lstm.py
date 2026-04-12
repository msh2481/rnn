import torch.nn as nn
from beartype import beartype
from einops import rearrange
from jaxtyping import Float
from torch import Tensor as TT

from src.dl.dl_base import DLBase, typed


class LSTM(DLBase):
    @beartype
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        **kw,
    ):
        super().__init__(**kw)
        self.input_dim = input_dim * self.mimo
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.readout = nn.Linear(hidden_dim, self.input_dim)

        self._h: tuple[TT, TT] | None = None

    @typed
    def forward(self, x: Float[TT, "B T D"]) -> Float[TT, "B T D"]:
        out, _ = self.lstm(x)
        return self.readout(out)

    @typed
    def predict_raw(self, x_t: Float[TT, "D"]) -> Float[TT, "D"]:
        x = rearrange(x_t, "D -> 1 1 D")
        out, self._h = self.lstm(x, self._h)
        pred = self.readout(out)
        return rearrange(pred, "1 1 D -> D")

    def reset_state(self) -> None:
        self._h = None
