"""Pueue-based oracle for BO hyperparameter optimization.

Shared machinery: normalization, script generation, pueue launch/wait/parse, CSV logging.
Domain-specific scripts provide the param specs, architecture template, and warm-start data.
"""

import csv
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from src.optim.bo import BO


def to_normalized(actual: np.ndarray, specs: list[tuple]) -> np.ndarray:
    out = np.empty(len(specs))
    for i, (_, low, high, log) in enumerate(specs):
        if log:
            out[i] = 2 * (np.log(actual[i]) - np.log(low)) / (np.log(high) - np.log(low)) - 1
        else:
            out[i] = 2 * (actual[i] - low) / (high - low) - 1
    return out


def from_normalized(x_norm: np.ndarray, specs: list[tuple]) -> dict:
    params = {}
    for i, (name, low, high, log) in enumerate(specs):
        t = np.clip((x_norm[i] + 1) / 2, 0, 1)
        if log:
            params[name] = np.exp(np.log(low) + t * (np.log(high) - np.log(low)))
        else:
            params[name] = low + t * (high - low)
    return params


def fmt_params(params: dict) -> str:
    return "  ".join(f"{k}={v:.5g}" for k, v in params.items())


def launch_pueue(script: Path) -> int:
    result = subprocess.run(["pueue", "add", "python", str(script)], capture_output=True, text=True)
    match = re.search(r"id (\d+)", result.stdout)
    return int(match.group(1))


def wait_pueue():
    subprocess.run(["pueue", "wait"], capture_output=True)


def parse_result(pueue_id: int) -> float | None:
    result = subprocess.run(["pueue", "log", str(pueue_id)], capture_output=True, text=True)
    matches = re.findall(r"val_r2=([\-\d.]+)", result.stdout)
    return float(matches[-1]) if matches else None


def log_result(path: str, name: str, r2: float):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, f"{r2:.6f}"])


def run_bo(
    *,
    arch_name: str,
    specs: list[tuple],
    x0_actual: np.ndarray,
    warmstart: list[tuple[dict, float]],
    make_script: callable,  # (name: str, params: dict) -> Path
    n_gens: int = 10,
    pop: int = 2,
    sigma: float = 0.3,
    seed: int = 42,
    results_file: str = "opt_results.csv",
):
    x0 = to_normalized(x0_actual, specs)
    opt = BO(x0=x0, sigma=sigma, seed=seed)

    # warm-start
    for params, r2 in warmstart:
        actual = np.array([params[name] for name, *_ in specs])
        x = to_normalized(actual, specs)
        opt.obs_x.append(x)
        opt.obs_y.append(r2)
        if r2 > opt.best_score:
            opt.best_score = r2
            opt.best_x = x.copy()
    print(f"BO for {arch_name}: {n_gens} gens, pop={pop}, warm-start={len(warmstart)} points, best={opt.best_score:.4f}")

    for gen in range(n_gens):
        xs = opt.ask(pop)
        configs = [from_normalized(x, specs) for x in xs]

        ids = []
        for i, cfg in enumerate(configs):
            name = f"bo_{arch_name}_g{gen}_{i}"
            script = make_script(name, cfg)
            pid = launch_pueue(script)
            ids.append((pid, name, cfg))
            print(f"  launched {name} (pueue {pid}): {fmt_params(cfg)}")

        wait_pueue()

        scores = []
        for pid, name, cfg in ids:
            r2 = parse_result(pid)
            if r2 is None:
                print(f"  WARNING: no result for {name} (pueue {pid}), using -1")
                r2 = -1.0
            scores.append(r2)
            log_result(results_file, name, r2)
            print(f"  {name}: r2={r2:.4f}")

        opt.tell(xs, np.array(scores))
        subprocess.run(["pueue", "clean"], capture_output=True)

        best_params = from_normalized(opt.best(), specs)
        print(f"  gen {gen}: best_so_far={opt.best_score:.4f}  {fmt_params(best_params)}\n")

    return opt
