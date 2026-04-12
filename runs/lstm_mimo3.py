from functools import partial
import torch.optim as optim
from src.dl import LSTM

LSTM(name="lstm_mimo3", input_dim=32, hidden_dim=128, num_layers=2, dropout=0.1,
     n_epochs=50, batch_size=16, mimo=3, optimizer_fn=partial(optim.Adam, lr=1e-3),
     scheduler_fn=None).fit()
