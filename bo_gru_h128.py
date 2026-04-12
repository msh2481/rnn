"""BO for GRU h128 l2."""

import json
from pathlib import Path

from scipy import stats

from src.optim.oracle import run_bo
from src.optim.params import Param

PARAMS = [
    Param("lr", stats.lognorm(s=1.0, scale=1e-3)),
    Param("beta1", stats.beta(15, 2)),
    Param("beta2", stats.beta(30, 1.2)),
    Param("weight_decay", stats.beta(1.5, 10)),
    Param("weight_drop", stats.beta(1.5, 10)),
    Param("output_drop", stats.beta(1.5, 10)),
]

X0 = dict(lr=1e-3, beta1=0.9, beta2=0.99, weight_decay=0.01, weight_drop=0.01, output_drop=0.01)

# Known results from manual experiments (all h128 l2, Adam, 50 epochs)
WARMSTART = [
    # best_repro: Adam lr=1e-3, default betas=(0.9, 0.999), no gc
    (dict(lr=1e-3,   beta1=0.9,  beta2=0.999, weight_decay=0.01, weight_drop=0.01, output_drop=0.01),  0.363),
    # best_gc: same + grad_clip=1.0
    (dict(lr=1e-3,   beta1=0.9,  beta2=0.999, weight_decay=0.01, weight_drop=0.01, output_drop=0.01),  0.364),
    # best_gc_b99: gc + betas=(0.9, 0.99)
    (dict(lr=1e-3,   beta1=0.9,  beta2=0.99,  weight_decay=0.01, weight_drop=0.01, output_drop=0.01),  0.365),
    # from results.csv: GRU h128 l2 e50 Adam lr=1e-3 (default betas)
    (dict(lr=1e-3,   beta1=0.9,  beta2=0.999, weight_decay=0.01, weight_drop=0.01, output_drop=0.01),  0.364),
    # dropout runs
    (dict(lr=3e-3,   beta1=0.9,  beta2=0.99,  weight_decay=0.3,  weight_drop=0.01, output_drop=0.3),   0.358),
    (dict(lr=3e-3,   beta1=0.9,  beta2=0.99,  weight_decay=0.1,  weight_drop=0.01, output_drop=0.01),  0.356),
]


def _load_jsonl_warmstart():
    """Load previous BO runs from jsonl, adding beta1=0.9 (old oracle default)."""
    path = Path("bo_h128l2.jsonl")
    if not path.exists():
        return []
    extra = []
    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        p = entry["params"]
        r2 = entry["r2"]
        if "beta1" not in p:
            p["beta1"] = 0.9
        extra.append((p, r2))
    return extra


def make_script(name: str, cfg: dict) -> Path:
    path = Path(f"runs/bo_{name}.py")
    path.write_text(f"""\
from functools import partial
import torch.optim as optim
from src.dl import GRU
GRU(name="{name}", input_dim=32, hidden_dim=128, num_layers=2,
    weight_drop={cfg['weight_drop']:.6f}, input_drop=0.0, output_drop={cfg['output_drop']:.6f},
    layer_drop=0.0, ortho_init=True,
    n_epochs=50, batch_size=16, mimo=1,
    optimizer_fn=partial(optim.Adam, lr={cfg['lr']:.6f}, betas=({cfg['beta1']:.6f}, {cfg['beta2']:.6f})),
    scheduler_fn=None, grad_clip=1.0).fit()
""")
    return path


if __name__ == "__main__":
    warmstart = WARMSTART + _load_jsonl_warmstart()
    run_bo(arch_name="h128l2", params=PARAMS, x0=X0, warmstart=warmstart, make_script=make_script)
