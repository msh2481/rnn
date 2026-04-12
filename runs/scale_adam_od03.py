from functools import partial
import torch.optim as optim
from src.dl import GRU
GRU(name="scale_adam_od03", input_dim=32, hidden_dim=128, num_layers=2, weight_drop=0.3, input_drop=0.0, output_drop=0.3, layer_drop=0.0,
    n_epochs=100, batch_size=16, mimo=1, optimizer_fn=partial(optim.Adam, lr=2.5e-3, betas=(0.9, 0.99)), scheduler_fn=None, grad_clip=1.0).fit()
