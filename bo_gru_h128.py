"""BO for GRU h128 l2."""

from pathlib import Path

import numpy as np

from src.optim.oracle import run_bo

SPECS = [
    # (name, low, high, log)
    ("lr",           1e-4,  1e-2,  True),
    ("beta2",        0.9,   0.999, False),
    ("weight_decay", 0.0,   0.5,   False),
    ("weight_drop",  0.0,   0.5,   False),
    ("output_drop",  0.0,   0.5,   False),
]

X0 = np.array([1e-3, 0.99, 0.0, 0.0, 0.0])

WARMSTART = [
    (dict(lr=1e-3,   beta2=0.99,  weight_decay=0.0, weight_drop=0.0, output_drop=0.0),   0.365),
    (dict(lr=1e-3,   beta2=0.999, weight_decay=0.0, weight_drop=0.0, output_drop=0.0),   0.364),
    (dict(lr=1e-3,   beta2=0.99,  weight_decay=0.0, weight_drop=0.0, output_drop=0.0),   0.363),
    (dict(lr=3e-3,   beta2=0.99,  weight_decay=0.3, weight_drop=0.0, output_drop=0.3),   0.358),
    (dict(lr=3e-3,   beta2=0.99,  weight_decay=0.1, weight_drop=0.0, output_drop=0.0),   0.356),
]


def make_script(name: str, params: dict) -> Path:
    path = Path(f"runs/bo_{name}.py")
    path.write_text(f"""\
from functools import partial
import torch.optim as optim
from src.dl import GRU
GRU(name="{name}", input_dim=32, hidden_dim=128, num_layers=2,
    weight_drop={params['weight_drop']:.6f}, input_drop=0.0, output_drop={params['output_drop']:.6f},
    layer_drop=0.0, ortho_init=True,
    n_epochs=50, batch_size=16, mimo=1,
    optimizer_fn=partial(optim.Adam, lr={params['lr']:.6f}, betas=(0.9, {params['beta2']:.6f})),
    scheduler_fn=None, grad_clip=1.0).fit()
""")
    return path


if __name__ == "__main__":
    run_bo(arch_name="h128l2", specs=SPECS, x0_actual=X0, warmstart=WARMSTART, make_script=make_script)
