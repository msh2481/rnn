"""BO for GRU h128 l2. Priors fitted from top-25% of previous BO runs, relaxed."""

from pathlib import Path

from scipy import stats

from src.optim.oracle import run_bo
from src.optim.params import Param, preview_priors

PARAMS = [
    Param("lr",           stats.lognorm(s=1.0, scale=2e-3)),     # median~2e-3, wide
    Param("beta1",        stats.beta(10, 2)),                     # mode~0.9, relaxed
    Param("beta2",        stats.beta(30, 1.2)),                   # mode~0.99, relaxed
    Param("weight_decay", stats.beta(0.4, 2.5)),                  # wide, lower center
    Param("weight_drop",  stats.beta(0.8, 8)),                    # small but wider
    Param("output_drop",  stats.beta(0.7, 20)),                   # small but wider
]

X0 = dict(lr=1e-3, beta1=0.9, beta2=0.99, weight_decay=0.05, weight_drop=0.01, output_drop=0.01)


def make_script(name: str, cfg: dict) -> Path:
    path = Path(f"runs/bo_{name}.py")
    path.write_text(f"""\
from functools import partial
import torch.optim as optim
from src.dl import GRU
GRU(name="{name}", input_dim=32, hidden_dim=128, num_layers=2,
    weight_drop={cfg['weight_drop']:.6f}, input_drop=0.0, output_drop={cfg['output_drop']:.6f},
    layer_drop=0.0, ortho_init=True,
    n_epochs=100, batch_size=16, mimo=1,
    optimizer_fn=partial(optim.Adam, lr={cfg['lr']:.6f}, betas=({cfg['beta1']:.6f}, {cfg['beta2']:.6f})),
    scheduler_fn=None, grad_clip=1.0).fit()
""")
    return path


if __name__ == "__main__":
    preview_priors(PARAMS)
    print()
    run_bo(arch_name="h128l2", params=PARAMS, x0=X0, warmstart=[], make_script=make_script)
