"""Test SkoptBO (with EIps) vs SampleBO on synthetic HP landscape."""

import numpy as np
from scipy import stats

from src.optim import SampleBO, SkoptBO
from src.optim.params import Param

# Same landscape as test_optim but using Param priors + probit space
PARAM_SPECS = [
    Param("lr", stats.lognorm(s=1.0, scale=1e-3)),
    Param("beta2", stats.beta(30, 1.2)),
    Param("weight_decay", stats.beta(1.5, 10)),
    Param("weight_drop", stats.beta(1.5, 10)),
    Param("output_drop", stats.beta(1.5, 10)),
]

OPT = dict(lr=1e-3, beta2=0.99, weight_decay=0.05, weight_drop=0.1, output_drop=0.05)
PEAK_SCORE = 0.370
X0 = dict(lr=1e-3, beta2=0.95, weight_decay=0.01, weight_drop=0.01, output_drop=0.01)


def oracle(config: dict, noise_std=0.005, rng=None):
    score = PEAK_SCORE
    d_lr = np.log(config["lr"]) - np.log(OPT["lr"])
    d_b2 = config["beta2"] - OPT["beta2"]
    d_wd = config["weight_decay"] - OPT["weight_decay"]
    d_wdr = config["weight_drop"] - OPT["weight_drop"]
    d_odr = config["output_drop"] - OPT["output_drop"]
    score -= 3.0 * d_lr ** 2
    score -= 0.5 * d_b2 ** 2 / 0.01
    score -= 0.3 * d_wd ** 2
    score -= 0.4 * d_wdr ** 2
    score -= 0.4 * d_odr ** 2
    score -= 1.5 * d_lr * (d_wdr + d_odr)
    total_drop = config["weight_drop"] + config["output_drop"]
    if total_drop > 0.3:
        score -= 2.0 * (total_drop - 0.3) ** 2
    if rng is not None:
        score += rng.normal(0, noise_std)
    return score


def sim_time(config: dict, rng=None) -> float:
    """Simulate training time: ~300s baseline, smaller models faster."""
    base = 300
    # higher dropout → slightly faster (less compute)
    base *= 1 - 0.1 * (config["weight_drop"] + config["output_drop"])
    if rng is not None:
        base *= rng.uniform(0.9, 1.1)
    return max(base, 60)


def run_sample_bo(n_gens=20, pop=2, noise_std=0.005, seed=0):
    rng = np.random.default_rng(seed + 1000)
    z0 = np.array([p.to_probit(X0[p.name]) for p in PARAM_SPECS])
    opt = SampleBO(x0=z0, sigma=1.0, seed=seed)
    for _ in range(n_gens):
        zs = opt.ask(pop)
        configs = [{p.name: p.from_probit(float(z[i])) for i, p in enumerate(PARAM_SPECS)} for z in zs]
        scores = np.array([oracle(c, noise_std, rng) for c in configs])
        opt.tell(zs, scores)
    best_config = {p.name: p.from_probit(float(opt.best()[i])) for i, p in enumerate(PARAM_SPECS)}
    return opt.best_score, best_config


def run_skopt(acq_func="EI", n_gens=20, pop=2, noise_std=0.005, seed=0):
    rng = np.random.default_rng(seed + 1000)
    opt = SkoptBO(PARAM_SPECS, X0, acq_func=acq_func, n_initial=4, seed=seed)
    for _ in range(n_gens):
        configs = opt.ask(pop)
        scores = [oracle(c, noise_std, rng) for c in configs]
        if acq_func == "EIps":
            times = [sim_time(c, rng) for c in configs]
            opt.tell(configs, scores, times)
        else:
            opt.tell(configs, scores)
    return opt.best_score, opt.best()


def eval_method(label, run_fn, n_seeds=30, **kw):
    scores = []
    for seed in range(n_seeds):
        s, _ = run_fn(seed=seed, **kw)
        scores.append(s)
    scores = np.array(scores)
    print(f"  {label:35s}  mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"min={scores.min():.4f}  max={scores.max():.4f}")


if __name__ == "__main__":
    print(f"Optimum: score={PEAK_SCORE:.4f}")
    print(f"Start:   score={oracle(X0, noise_std=0):.4f}\n")

    print("=== 40 evals (pop=2, 20 gens) ===")
    eval_method("SampleBO (Thompson)", run_sample_bo, n_seeds=10, pop=2, n_gens=20)
    eval_method("SkoptBO (EI)", run_skopt, n_seeds=10, acq_func="EI", pop=2, n_gens=20)
