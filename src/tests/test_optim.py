"""Test BO (new version) vs old on synthetic HP landscape."""

import numpy as np

from src.optim import BO, SampleBO

PARAMS = [
    ("log_lr",      np.log(1e-3),   np.log(10)),
    ("beta2",       0.95,           0.05),
    ("weight_decay", 0.25,          0.25),
    ("weight_drop", 0.25,           0.25),
    ("output_drop", 0.25,           0.25),
]
CENTERS = np.array([p[1] for p in PARAMS])
SCALES = np.array([p[2] for p in PARAMS])

OPT_ACTUAL = np.array([np.log(1e-3), 0.99, 0.05, 0.1, 0.05])
OPT_NORM = (OPT_ACTUAL - CENTERS) / SCALES
PEAK_SCORE = 0.370
LOG_LR, BETA2, WD, WDROP, ODROP = range(5)


def to_actual(x_norm):
    return CENTERS + x_norm * SCALES


def oracle(x_norm, noise_std=0.005, rng=None):
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


def fmt(x_norm):
    x = to_actual(x_norm)
    return (f"lr={np.exp(x[LOG_LR]):.5f} beta2={x[BETA2]:.4f} "
            f"wd={x[WD]:.4f} wdrop={x[WDROP]:.4f} odrop={x[ODROP]:.4f}")


def x0_norm():
    x0_actual = np.array([np.log(1e-3), 0.95, 0.0, 0.0, 0.0])
    return (x0_actual - CENTERS) / SCALES


def run_bo(n_gens=10, pop=2, sigma=0.3, noise_std=0.005, seed=0):
    rng = np.random.default_rng(seed + 1000)
    opt = BO(x0=x0_norm(), sigma=sigma, seed=seed)
    for _ in range(n_gens):
        xs = opt.ask(pop)
        scores = np.array([oracle(xi, noise_std, rng) for xi in xs])
        opt.tell(xs, scores)
    return opt


def eval_bo(label, n_seeds=50, **kw):
    scores = []
    for seed in range(n_seeds):
        opt = run_bo(seed=seed, **kw)
        scores.append(opt.best_score)
    scores = np.array(scores)
    print(f"  {label:30s}  mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"min={scores.min():.4f}  max={scores.max():.4f}")


if __name__ == "__main__":
    print(f"Optimum: {fmt(OPT_NORM)} score={PEAK_SCORE:.4f}")
    print(f"Start:   {fmt(x0_norm())} score={oracle(x0_norm(), noise_std=0):.4f}\n")

    far_x0 = np.array([-1.5, -1.5, 1.5, 1.5, 1.5])

    def run_bo_x0(x0, n_gens=10, pop=2, sigma=0.3, noise_std=0.005, seed=0, **kw):
        rng = np.random.default_rng(seed + 1000)
        opt = BO(x0=x0, sigma=sigma, seed=seed, **kw)
        for _ in range(n_gens):
            xs = opt.ask(pop)
            scores = np.array([oracle(xi, noise_std, rng) for xi in xs])
            opt.tell(xs, scores)
        return opt

    N_SEEDS = 10

    def sweep(label, x0, n_seeds=N_SEEDS, **kw):
        scores = []
        for seed in range(n_seeds):
            opt = run_bo_x0(x0, seed=seed, **kw)
            scores.append(opt.best_score)
        scores = np.array(scores)
        print(f"  {label:35s}  mean={scores.mean():.4f}  std={scores.std():.4f}  "
              f"min={scores.min():.4f}  max={scores.max():.4f}")

    for kappa in [0.1, 1.0]:
        print(f"=== BO  kappa={kappa} ===")
        print(f"  Close start (score={oracle(x0_norm(), noise_std=0):.3f}):")
        for pop, gens in [(1, 40), (2, 20), (4, 10)]:
            sweep(f"  pop={pop} gens={gens}", x0_norm(), pop=pop, n_gens=gens, ucb_kappa=kappa)
        print(f"  Far start (score={oracle(far_x0, noise_std=0):.3f}):")
        for pop, gens in [(1, 40), (2, 20), (4, 10)]:
            sweep(f"  pop={pop} gens={gens}", far_x0, pop=pop, n_gens=gens, sigma=0.3, ucb_kappa=kappa)
        print()

    # --- SampleBO (Thompson sampling) ---
    def sweep_sample(label, x0, n_seeds=N_SEEDS, n_gens=10, pop=2,
                     sigma=0.3, noise_std=0.005, **kw):
        scores = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 1000)
            opt = SampleBO(x0=x0, sigma=sigma, seed=seed, **kw)
            for _ in range(n_gens):
                xs = opt.ask(pop)
                sc = np.array([oracle(xi, noise_std, rng) for xi in xs])
                opt.tell(xs, sc)
            scores.append(opt.best_score)
        scores = np.array(scores)
        print(f"  {label:35s}  mean={scores.mean():.4f}  std={scores.std():.4f}  "
              f"min={scores.min():.4f}  max={scores.max():.4f}")

    for temp in [1.0, 2.0, 5.0]:
        print(f"=== SampleBO (Thompson)  temp={temp} ===")
        print(f"  Close start (score={oracle(x0_norm(), noise_std=0):.3f}):")
        for pop, gens in [(1, 40), (2, 20), (4, 10)]:
            sweep_sample(f"  pop={pop} gens={gens}", x0_norm(), pop=pop, n_gens=gens, temperature=temp)
        print(f"  Far start (score={oracle(far_x0, noise_std=0):.3f}):")
        for pop, gens in [(1, 40), (2, 20), (4, 10)]:
            sweep_sample(f"  pop={pop} gens={gens}", far_x0, pop=pop, n_gens=gens, sigma=0.3, temperature=temp)
        print()
