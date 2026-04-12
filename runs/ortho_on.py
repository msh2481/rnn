from functools import partial
from src.dl import GRU
from src.dl.optim import MuonAdam
GRU(name="ortho_on", input_dim=32, hidden_dim=128, num_layers=2, weight_drop=0.3, input_drop=0.0, output_drop=0.3, layer_drop=0.0, ortho_init=True,
    n_epochs=100, batch_size=16, mimo=1, optimizer_fn=partial(MuonAdam, lr=0.005), scheduler_fn=None, grad_clip=1.0).fit()
