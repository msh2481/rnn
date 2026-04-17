"""Non-monotonically Triggered ASGD (Merity et al., 2017).

After val loss fails to improve for `patience` epochs, starts maintaining
a running average of parameters. Use as context manager to temporarily
swap in averaged params for evaluation.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager


class NTASGD:
    def __init__(self, model: nn.Module, patience: int = 5):
        self.model = model
        self.patience = patience
        self.best_val_loss = float("inf")
        self.stale_epochs = 0
        self.triggered = False
        self.n_averaged = 0
        self._avg_params: dict[str, torch.Tensor] | None = None

    def force_trigger(self):
        if self.triggered:
            return
        self.triggered = True
        self._avg_params = {
            name: p.data.clone() for name, p in self.model.named_parameters()
        }
        self.n_averaged = 1

    def avg_step(self, val_loss: float):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1

        if not self.triggered and self.stale_epochs >= self.patience:
            self.triggered = True
            self._avg_params = {
                name: p.data.clone() for name, p in self.model.named_parameters()
            }
            self.n_averaged = 1

        elif self.triggered:
            self.n_averaged += 1
            for name, p in self.model.named_parameters():
                self._avg_params[name] += (p.data - self._avg_params[name]) / self.n_averaged

    @contextmanager
    def averaged(self):
        if self._avg_params is None:
            yield
            return
        backup = {name: p.data.clone() for name, p in self.model.named_parameters()}
        for name, p in self.model.named_parameters():
            p.data.copy_(self._avg_params[name])
        try:
            yield
        finally:
            for name, p in self.model.named_parameters():
                p.data.copy_(backup[name])

    def swap_in_averaged(self):
        if self._avg_params is None:
            return
        for name, p in self.model.named_parameters():
            p.data.copy_(self._avg_params[name])
