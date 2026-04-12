"""Test NES and CMA-ES on a synthetic HP optimization landscape.

The oracle works in normalized coordinates (each param mapped to ~unit scale).
Optimizers see a well-conditioned space where a single sigma makes sense.
"""

import numpy as np
import cma

from src.optim import NES, BO

# Param specs: (name, center, half_range)
# Optimizer sees normalized coords: actual = center + x * half_range
PARAMS = [
    ("log_lr",      np.log(1e-3),   np.log(10)),  # 1e-4 .. 1e-2
    ("beta2",       0.95,           0.05),          # 0.9 .. 1.0
    ("weight_decay", 0.25,          0.25),          # 0.0 .. 0.5
    ("weight_drop", 0.25,           0.25),          # 0.0 .. 0.5
    ("output_drop", 0.25,           0.25),          # 0.0 .. 0.5
]
NAMES = [p[0] for p in PARAMS]
CENTERS = np.array([p[1] for p in PARAMS])
SCALES = np.array([p[2] for p in PARAMS])

# Optimum in actual space
OPT_ACTUAL = np.array([np.log(1e-3), 0.99, 0.05, 0.1, 0.05])
# Optimum in normalized space
OPT_NORM = (OPT_ACTUAL - CENTERS) / SCALES

PEAK_SCORE = 0.370
LOG_LR, BETA2, WD, WDROP, ODROP = range(5)


def to_actual(x_norm: np.ndarray) -> np.ndarray:
    return CENTERS + x_norm * SCALES


def oracle(x_norm: np.ndarray, noise_std: float = 0.005, rng: np.random.Generator | None = None) -> float:
    x = to_actual(x_norm)
    d = x - OPT_ACTUAL

    score = PEAK_SCORE
    score -= 3.0 * d[LOG_LR] ** 2
    score -= 0.5 * d[BETA2] ** 2 / 0.01
    score -= 0.3 * d[WD] ** 2
    score -= 0.4 * d[WDROP] ** 2
    score -= 0.4 * d[ODROP] ** 2
    score -= 1.5 * d[LOG_LR] * (d[WDROP] + d[ODROP])

    total_drop = x[WDROP] + x[ODROP]
    if total_drop > 0.3:
        score -= 2.0 * (total_drop - 0.3) ** 2

    if rng is not None:
        score += rng.normal(0, noise_std)
    return score


def fmt(x_norm: np.ndarray) -> str:
    x = to_actual(x_norm)
    return (f"lr={np.exp(x[LOG_LR]):.5f} beta2={x[BETA2]:.4f} "
            f"wd={x[WD]:.4f} wdrop={x[WDROP]:.4f} odrop={x[ODROP]:.4f}")


def x0_norm():
    """Starting point in normalized space (our current best guess, slightly off)."""
    x0_actual = np.array([np.log(1e-3), 0.95, 0.0, 0.0, 0.0])
    return (x0_actual - CENTERS) / SCALES


def run_nes(n_gens=10, pop=2, sigma=0.1, lr=0.5, noise_std=0.005, seed=0, x0_override=None):
    rng = np.random.default_rng(seed + 1000)
    start = x0_override if x0_override is not None else x0_norm()
    opt = NES(x0=start, sigma=sigma, lr=lr, seed=seed)
    for _ in range(n_gens):
        xs = opt.ask(pop)
        scores = np.array([oracle(xi, noise_std, rng) for xi in xs])
        opt.tell(xs, scores)
    return opt


def run_bo(x0, n_gens=10, pop=2, sigma=0.1, noise_std=0.005, seed=0):
    rng = np.random.default_rng(seed + 1000)
    opt = BO(x0=x0, sigma=sigma, seed=seed)
    for _ in range(n_gens):
        xs = opt.ask(pop)
        scores = np.array([oracle(xi, noise_std, rng) for xi in xs])
        opt.tell(xs, scores)
    return opt


def eval_bo(label, x0, n_seeds=20, **kw):
    scores = []
    for seed in range(n_seeds):
        opt = run_bo(x0=x0, seed=seed, **kw)
        scores.append(opt.best_score)
    scores = np.array(scores)
    print(f"  {label:30s}  mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"min={scores.min():.4f}  max={scores.max():.4f}")
    return scores.mean()


def eval_config(label, n_seeds=20, x0_override=None, **kw):
    scores = []
    for seed in range(n_seeds):
        opt = run_nes(seed=seed, x0_override=x0_override, **kw)
        scores.append(opt.best_score)
    scores = np.array(scores)
    print(f"  {label:30s}  mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"min={scores.min():.4f}  max={scores.max():.4f}")
    return scores.mean()


if __name__ == "__main__":
    print(f"Optimum: {fmt(OPT_NORM)} score={PEAK_SCORE:.4f}")
    print(f"Start:   {fmt(x0_norm())} score={oracle(x0_norm(), noise_std=0):.4f}")
    print()

    n_seeds = 50
    n_gens = 10

    print(f"=== BO parallelism: {n_gens} gens, fixed budget=40 evals, close start ===")
    for pop in [1, 2, 4, 8]:
        gens = 40 // pop
        eval_bo(f"pop={pop} gens={gens}", x0=x0_norm(), sigma=0.1, n_gens=gens, pop=pop, n_seeds=n_seeds)

    print(f"\n=== BO parallelism: {n_gens} gens, fixed budget=40 evals, far start ===")
    far_x0 = np.array([-1.5, -1.5, 1.5, 1.5, 1.5])
    print(f"    start: {fmt(far_x0)} score={oracle(far_x0, noise_std=0):.4f}")
    for pop in [1, 2, 4, 8]:
        gens = 40 // pop
        eval_bo(f"pop={pop} gens={gens}", x0=far_x0, sigma=0.3, n_gens=gens, pop=pop, n_seeds=n_seeds)

    print(f"\n=== NES baseline (same budget) ===")
    eval_config("NES close lr=0.03", sigma=0.1, lr=0.03, n_gens=20, n_seeds=n_seeds)
    eval_config("NES far lr=0.01", sigma=0.1, lr=0.01, n_gens=20, n_seeds=n_seeds, x0_override=far_x0)
