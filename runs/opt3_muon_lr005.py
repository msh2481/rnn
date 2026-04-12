from functools import partial
from src.dl import GRU
from src.dl.optim import MuonAdam
GRU(name="opt3_muon_lr005", input_dim=32, hidden_dim=64, num_layers=1, weight_drop=0.0, input_drop=0.0, output_drop=0.0, layer_drop=0.0,
    n_epochs=50, batch_size=16, mimo=1, optimizer_fn=partial(MuonAdam, lr=0.005), scheduler_fn=None, grad_clip=1.0).fit()
