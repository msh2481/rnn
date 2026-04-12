from functools import partial
import torch.optim as optim
from src.dl import GRU
GRU(name="opt2_adam_lr1e3_b85_b99", input_dim=32, hidden_dim=64, num_layers=1, weight_drop=0.0, input_drop=0.0, output_drop=0.0, layer_drop=0.0,
    n_epochs=30, batch_size=16, mimo=1, optimizer_fn=partial(optim.Adam, lr=1e-3, betas=(0.85, 0.99)), scheduler_fn=None, grad_clip=1.0).fit()
