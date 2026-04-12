from functools import partial
from src.dl import GRU
from src.dl.optim import MuonAdam
GRU(name="h64_l3_drop03", input_dim=32, hidden_dim=64, num_layers=3, weight_drop=0.3, input_drop=0.0, output_drop=0.3, layer_drop=0.0, ortho_init=False,
    n_epochs=100, batch_size=16, mimo=1, optimizer_fn=partial(MuonAdam, lr=0.005), scheduler_fn=None, grad_clip=1.0).fit()
