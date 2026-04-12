from functools import partial
from src.dl import GRU
from src.dl.optim import MuonAdam
GRU(name="mimo2_b16_drop0", input_dim=32, hidden_dim=128, num_layers=2, weight_drop=0.0, input_drop=0.0, output_drop=0.0, layer_drop=0.0, ortho_init=False,
    n_epochs=300, batch_size=16, mimo=2, optimizer_fn=partial(MuonAdam, lr=0.005), scheduler_fn=None, grad_clip=1.0).fit()
