from functools import partial
import torch.optim as optim
from src.dl import GRU

GRU(name="gru_cos3e3_wd01_ld01", input_dim=32, hidden_dim=128, num_layers=2, weight_drop=0.1, input_drop=0.0, output_drop=0.0, layer_drop=0.1,
    n_epochs=100, batch_size=16, mimo=1, optimizer_fn=partial(optim.Adam, lr=3e-3),
    scheduler_fn=partial(optim.lr_scheduler.CosineAnnealingLR, T_max=100)).fit()
