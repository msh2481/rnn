from functools import partial
import torch.optim as optim
from src.dl import LSTM

LSTM(name="lstm_wd03", input_dim=32, hidden_dim=128, num_layers=2, weight_drop=0.3, input_drop=0.0, output_drop=0.0,
     n_epochs=50, batch_size=16, mimo=1, optimizer_fn=partial(optim.Adam, lr=1e-3), scheduler_fn=None).fit()
