from contextlib import contextmanager

import torch
import torch.optim as optim


class MuonAdam(optim.Optimizer):
    """Muon for 2D+ params, Adam for the rest. With NT-ASGD-style parameter averaging.

    Averaging is triggered when val loss stalls for `patience` epochs,
    or force-triggered at 75% of training. Use `averaged()` context manager
    to temporarily swap in averaged params for evaluation.
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, adam_lr=1e-3, weight_decay=0.1, patience=5):
        params = list(params)
        muon_params = [p for p in params if p.dim() >= 2]
        adam_params = [p for p in params if p.dim() < 2]

        defaults = dict(lr=lr, momentum=momentum, adam_lr=adam_lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.param_groups.clear()
        self._optims = []
        if muon_params:
            muon = optim.Muon(muon_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
            self._optims.append(muon)
            self.param_groups.extend(muon.param_groups)
        if adam_params:
            adam = optim.Adam(adam_params, lr=adam_lr)
            self._optims.append(adam)
            self.param_groups.extend(adam.param_groups)

        # averaging state
        self.patience = patience
        self.best_val_loss = float("inf")
        self.stale_epochs = 0
        self.triggered = False
        self.n_averaged = 0
        self._avg_params: dict[int, torch.Tensor] | None = None
        self._all_params = params

    @torch.no_grad()
    def step(self, closure=None):
        for o in self._optims:
            o.step(closure)

    def zero_grad(self, set_to_none=True):
        for o in self._optims:
            o.zero_grad(set_to_none=set_to_none)

    # --- averaging (same interface as NTASGD) ---

    def force_trigger(self):
        if self.triggered:
            return
        self.triggered = True
        self._avg_params = {id(p): p.data.clone() for p in self._all_params}
        self.n_averaged = 1

    def avg_step(self, val_loss: float):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1

        if not self.triggered and self.stale_epochs >= self.patience:
            self.triggered = True
            self._avg_params = {id(p): p.data.clone() for p in self._all_params}
            self.n_averaged = 1
        elif self.triggered:
            self.n_averaged += 1
            for p in self._all_params:
                self._avg_params[id(p)] += (p.data - self._avg_params[id(p)]) / self.n_averaged

    @contextmanager
    def averaged(self):
        if self._avg_params is None:
            yield
            return
        backup = {id(p): p.data.clone() for p in self._all_params}
        for p in self._all_params:
            p.data.copy_(self._avg_params[id(p)])
        try:
            yield
        finally:
            for p in self._all_params:
                p.data.copy_(backup[id(p)])

    def swap_in_averaged(self):
        if self._avg_params is None:
            return
        for p in self._all_params:
            p.data.copy_(self._avg_params[id(p)])
