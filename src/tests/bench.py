"""Benchmark: time 4-epoch runs on small.parquet, report mean +- std over 5 runs."""
import os
import time
from functools import partial

os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch
import torch.optim as optim

from src.dl import GRU, TCN

CONFIGS = {
    "tcn_h128_l6_k3": dict(cls=TCN, hidden_dim=128, num_layers=6, kernel_size=3, dropout=0.1),
    "tcn_shared": dict(cls=TCN, hidden_dim=128, num_layers=6, kernel_size=3, dropout=0.1, shared_middle=4),
}

SMALL = "datasets/small.parquet"
N_RUNS = 5
N_EPOCHS = 4


GRU_KW = dict(
    weight_drop=0.006, input_drop=0.0, output_drop=0.0,
    layer_drop=0.0, ortho_init=True,
    input_noise=0.1, aux_horizons=(4,), aux_weight=0.5,
)


def bench(name, cfg, device):
    cfg = dict(cfg)
    cls = cfg.pop("cls")
    extra = GRU_KW if cls is GRU else {}
    m = cls(
        name=name, input_dim=32, **cfg, **extra,
        n_epochs=N_EPOCHS, batch_size=16, mimo=1,
        optimizer_fn=partial(optim.SGD, lr=0.21, momentum=0.98),
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
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        print(f"\n=== Device: {device} ===")
        for name, cfg in CONFIGS.items():
            times = [bench(name, cfg, device) for _ in range(N_RUNS)]
            t = np.array(times)
            print(f"  {name:20s}  {t.mean():.3f} ± {t.std():.3f}s")
