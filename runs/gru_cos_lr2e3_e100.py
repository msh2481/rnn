from functools import partial
import torch.optim as optim
from src.dl import GRU

GRU(name="gru_cos_lr2e3_e100", input_dim=32, hidden_dim=128, num_layers=2, weight_drop=0.0, input_drop=0.0, output_drop=0.0, layer_drop=0.0,
    n_epochs=100, batch_size=16, mimo=1, optimizer_fn=partial(optim.Adam, lr=2e-3),
    scheduler_fn=partial(optim.lr_scheduler.CosineAnnealingLR, T_max=100)).fit()
