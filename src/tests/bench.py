"""Benchmark: time 4-epoch runs, report mean ± std over N runs. Uses small.parquet.

Auto-creates small.parquet from train.parquet's first 10 sequences if missing.
"""
import os
import time
from functools import partial

os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.dl import GRU, TCN, LayerSpec
from src.dl.reparam import solve_scale


SMALL = "datasets/small.parquet"
TRAIN = "datasets/train.parquet"
N_RUNS = 3
N_EPOCHS = 4


def ensure_small():
    if os.path.exists(SMALL):
        return
    df = pd.read_parquet(TRAIN)
    seq_col = df.columns[0]
    ids = sorted(df[seq_col].unique())[:10]
    sub = df[df[seq_col].isin(ids)].reset_index(drop=True)
    os.makedirs(os.path.dirname(SMALL) or ".", exist_ok=True)
    sub.to_parquet(SMALL)
    print(f"created {SMALL}: {sub.shape}")


def build_tcn_center():
    specs = [LayerSpec(3, 64)] * 2 + [LayerSpec(3, 64, n_rep=4)] + [LayerSpec(3, 64)] * 2
    _, scaled = solve_scale(specs, 80000, 32, 1)
    return dict(
        cls=TCN, input_dim=32, layers=scaled, groups=1, dropout=0.1,
        optimizer_fn=partial(optim.Adam, lr=3e-3),
    )


CONFIGS = {
    "gru_h64_l1": dict(
        cls=GRU, input_dim=32, hidden_dim=64, num_layers=1,
        weight_drop=0.01, input_drop=0.0, output_drop=0.0, layer_drop=0.0,
        ortho_init=True,
        optimizer_fn=partial(optim.Adam, lr=1e-3),
    ),
    "gru_h128_l2": dict(
        cls=GRU, input_dim=32, hidden_dim=128, num_layers=2,
        weight_drop=0.01, input_drop=0.0, output_drop=0.0, layer_drop=0.0,
        ortho_init=True,
        optimizer_fn=partial(optim.Adam, lr=1e-3),
    ),
    "tcn_center": build_tcn_center(),
}


def bench(name, cfg, device):
    cfg = dict(cfg)
    cls = cfg.pop("cls")
    m = cls(
        name=name, **cfg,
        n_epochs=N_EPOCHS, batch_size=16, mimo=1,
        scheduler_fn=None, grad_clip=1.0, asgd_patience=5,
    ).to(device)

    t0 = time.perf_counter()
    m.fit(dataset=SMALL, val_dataset=SMALL, show_progress=False)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


if __name__ == "__main__":
    ensure_small()
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        print(f"\n=== Device: {device} ===")
        for name, cfg in CONFIGS.items():
            times = [bench(name, cfg, device) for _ in range(N_RUNS)]
            t = np.array(times)
            print(f"  {name:20s}  {t.mean():.3f} ± {t.std():.3f}s   ({t.mean()/N_EPOCHS:.3f}s/epoch)")
