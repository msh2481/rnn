from functools import partial
import torch.optim as optim
from src.dl import LSTM

LSTM(name="lstm_h64_l1_e50", input_dim=32, hidden_dim=64, num_layers=1, dropout=0.0,
     n_epochs=50, batch_size=32, mimo=1, optimizer_fn=partial(optim.Adam, lr=1e-3), scheduler_fn=None).fit()
