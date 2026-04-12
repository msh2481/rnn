# Welcome to the Wunder Challenge!
2025-09-15

We're excited to have you here. This is a machine learning competition where you'll build a model to predict the future of market states from their past. It’s a tough challenge, but a very rewarding one. Let's get started!

## Your mission

Your goal is to predict the next market state vector based on the sequence of states that came before it. Think of it as a sequence modeling problem. You'll be given the market's history up to a certain point, and you need to forecast what happens next.

## How it works

The dataset is a single table in Parquet format, containing multiple independent sequences. Here’s what you need to know.

### The data format

Each row in the table represents a single market state at a specific step in a sequence. The table has **N + 3** columns:

*   `seq_ix`: An ID for the sequence. When this number changes, you're starting a new, completely independent sequence.
*   `step_in_seq`: The step number within a sequence (from 0 to 999).
*   `need_prediction`: A boolean that’s `True` if we need a prediction from you for the *next* step, and `False` otherwise.
*   **N feature columns**: The remaining `N` columns are the anonymized numeric features that describe the market state.

### The sequences

Each sequence is exactly **1000 steps** long.

> **Note:**
> The first 100 steps (0-99) of every sequence are for warm-up. Your model can use them to build context, but we won't score your predictions here. Your score comes from predictions for steps 100 to 998.

Because each sequence is independent, you must reset your model’s internal state whenever you see a new `seq_ix`.

You can also rely on two key facts about the data ordering:
*   **Within a sequence**, all steps are ordered by time.
*   **The sequences themselves** are randomly shuffled, so `seq_ix` and `seq_ix + 1` are not related.

> **Tip: How to create a validation set**
> Since all the sequences are independent and shuffled, you can create a reliable local validation set by splitting the sequences. For example, you could use the first 80% of the sequences for training and the remaining 20% for validation. You can split them by `seq_ix`.

## Evaluation and metrics

We'll evaluate your predictions using the **R²** (coefficient of determination) score.

For each feature *i*, the score is calculated as:
R²ᵢ = 1 - Σ(y_true - y_pred)² / Σ(y_true - y_mean)²

The final score is the average of the R² scores across all N features.

A higher R² score is better!

---

---

## Development guide

### Project structure

```
src/
├── utils.py              # DataPoint, Stack, Bag, ScorerStepByStep, train_and_eval
├── simple/               # Lightweight baselines (EMA, Poly, Mu100_3)
├── esn/                  # Echo State Networks (ESN, ES2N)
├── dl/                   # Deep learning models
│   ├── dl_base.py        # DLBase, SequenceDataset, W&B integration
│   └── lstm.py           # LSTM
└── tests/
    └── benchmark.py      # Dataloader benchmarks
```

### Running experiments

All commands run from the project root.

```bash
# Run LSTM baseline
python -m src.dl.lstm

# Or inline with custom params
python -c "
from functools import partial
import torch.optim as optim
from src.dl import LSTM

model = LSTM(
    name='lstm_h256_lr3e4',
    input_dim=32,
    hidden_dim=256,
    num_layers=2,
    dropout=0.1,
    n_epochs=30,
    batch_size=32,
    mimo=1,
    optimizer_fn=partial(optim.Adam, lr=3e-4),
    scheduler_fn=None,
)
model.fit()
"

# ESN models (non-DL, no W&B)
python -c "
from src.esn import ES2N
from src.utils import train_and_eval
train_and_eval(ES2N(reservoir_size=64))
"
```

### Naming convention

The `name` arg becomes the W&B run name (timestamp appended automatically):
- `lstm_h128_lr1e3` — hidden_dim=128, lr=1e-3
- `lstm_h256_l3_drop02` — hidden_dim=256, num_layers=3, dropout=0.2
- `lstm_mimo2_h128` — mimo=2, hidden_dim=128

### Pueue workflow

```bash
pueue add -- python -c "..."   # queue a job
pueue status                    # check progress
pueue wait                      # block until all jobs finish
pueue log <id>                  # read output of completed job
```

### W&B

- All DL runs log to project `rnn_challenge` by default
- Per batch: `train/batch_loss`
- Per epoch: `train/epoch_loss`, `val/loss`, `val/r2`
- Config auto-scraped from model attributes
- `WANDB_MODE=offline` for local-only runs
