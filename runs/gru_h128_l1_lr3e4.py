from functools import partial
import torch.optim as optim
from src.dl import GRU

GRU(name="gru_h128_l1_lr3e4", input_dim=32, hidden_dim=128, num_layers=1, weight_drop=0.0, input_drop=0.0, output_drop=0.0,
    n_epochs=50, batch_size=16, mimo=1, optimizer_fn=partial(optim.Adam, lr=3e-4), scheduler_fn=None).fit()
