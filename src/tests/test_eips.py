"""Test EI vs EIps on a problem where one param only affects speed.

obj(x, y) = -(x - 1.7)^2     (y doesn't affect score)
time(x, y) = 1 + 5 * y^2     (y affects time: y=0 is fast, y=±2 is slow)

EIps should learn to prefer y≈0 (fast) while EI wastes evals exploring y.
"""

import numpy as np
from scipy import stats

from src.optim import SkoptBO
from src.optim.params import Param

PARAMS = [
    Param("x", stats.uniform(loc=-3, scale=6)),   # uniform on [-3, 3]
    Param("y", stats.uniform(loc=-3, scale=6)),
]

X0 = dict(x=0.0, y=0.0)


def oracle(config, noise_std=0.01, rng=None):
    score = -(config["x"] - 1.7) ** 2
    if rng:
        score += rng.normal(0, noise_std)
    return score


def sim_time(config, rng=None):
    t = 1 + 5 * config["y"] ** 2
    if rng:
        t *= rng.uniform(0.9, 1.1)
    return t


def run(acq_func, n_evals=30, seed=0):
    rng = np.random.default_rng(seed + 1000)
    opt = SkoptBO(PARAMS, X0, acq_func=acq_func, n_initial=4, seed=seed)
    total_time = 0
    for _ in range(n_evals):
        configs = opt.ask(1)
        scores = [oracle(c, rng=rng) for c in configs]
        times = [sim_time(c, rng=rng) for c in configs]
        total_time += sum(times)
        if acq_func == "EIps":
            opt.tell(configs, scores, times)
        else:
            opt.tell(configs, scores)
    return opt.best_score, opt.best(), total_time


if __name__ == "__main__":
    print("Optimum: x=1.7, y=anything, score=0.0, fastest at y=0\n")

    n_seeds = 10
    for acq in ["EI", "EIps"]:
        scores, times, ys = [], [], []
        for seed in range(n_seeds):
            s, best, t = run(acq, seed=seed)
            scores.append(s)
            times.append(t)
            ys.append(abs(best["y"]))
        scores, times, ys = np.array(scores), np.array(times), np.array(ys)
        print(f"{acq:5s}  score={scores.mean():.4f}±{scores.std():.4f}"
              f"  time={times.mean():.0f}±{times.std():.0f}"
              f"  |y|={ys.mean():.2f}±{ys.std():.2f}")
