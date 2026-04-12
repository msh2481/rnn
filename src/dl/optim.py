import torch.optim as optim


class MuonAdam:
    """Muon for 2D params (weight matrices), Adam for the rest (biases)."""

    def __init__(self, params, lr=1e-3, adam_lr=1e-3, weight_decay=0.1):
        params = list(params)
        muon_params = [p for p in params if p.dim() >= 2]
        adam_params = [p for p in params if p.dim() < 2]
        self.muon = optim.Muon(muon_params, lr=lr, weight_decay=weight_decay) if muon_params else None
        self.adam = optim.Adam(adam_params, lr=adam_lr) if adam_params else None

    def zero_grad(self):
        if self.muon:
            self.muon.zero_grad()
        if self.adam:
            self.adam.zero_grad()

    def step(self):
        if self.muon:
            self.muon.step()
        if self.adam:
            self.adam.step()
