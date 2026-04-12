"""Pueue-based oracle for BO hyperparameter optimization.

Shared machinery: probit conversion, script generation, pueue launch/wait/parse, JSONL logging.
Domain-specific scripts provide the param priors, architecture template, and warm-start data.

Each BO instance uses its own pueue group to avoid interference.
"""

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from src.optim.bo import BO
from src.optim.params import Param


def to_probit(actual: dict, params: list[Param]) -> np.ndarray:
    return np.array([p.to_probit(actual[p.name]) for p in params])


def from_probit(z: np.ndarray, params: list[Param]) -> dict:
    return {p.name: p.from_probit(float(z[i])) for i, p in enumerate(params)}


def fmt_params(config: dict) -> str:
    return "  ".join(f"{k}={v:.5g}" for k, v in config.items())


def _pueue(args: list[str]) -> str:
    result = subprocess.run(["pueue"] + args, capture_output=True, text=True)
    return result.stdout


def setup_group(group: str):
    _pueue(["group", "add", group])
    _pueue(["parallel", "2", "--group", group])


def launch_pueue(script: Path, group: str) -> int:
    result = _pueue(["add", "--group", group, "python", str(script)])
    match = re.search(r"id (\d+)", result)
    return int(match.group(1))


def wait_pueue(group: str):
    subprocess.run(["pueue", "wait", "--group", group], capture_output=True)


def parse_result(pueue_id: int) -> float | None:
    result = _pueue(["log", str(pueue_id)])
    matches = re.findall(r"val_r2=([\-\d.]+)", result)
    return float(matches[-1]) if matches else None


def clean_pueue(group: str):
    _pueue(["clean", "--group", group])


def log_result(path: str, config: dict, r2: float):
    with open(path, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "params": config, "r2": r2}) + "\n")


def run_bo(
    *,
    arch_name: str,
    params: list[Param],
    x0: dict,
    warmstart: list[tuple[dict, float]],
    make_script: callable,  # (name: str, config: dict) -> Path
    pop: int = 2,
    sigma: float = 0.3,
    ucb_kappa: float = 0.1,
    seed: int = 42,
    results_file: str | None = None,
):
    if results_file is None:
        results_file = f"bo_{arch_name}.jsonl"

    group = f"bo_{arch_name}"
    setup_group(group)

    z0 = to_probit(x0, params)
    opt = BO(x0=z0, sigma=sigma, ucb_kappa=ucb_kappa, seed=seed)

    # warm-start (jitter probit coords slightly to break ties)
    jitter_rng = np.random.default_rng(seed)
    for config, r2 in warmstart:
        z = to_probit(config, params)
        z += jitter_rng.normal(0, 0.01, size=z.shape)
        opt.obs_x.append(z)
        opt.obs_y.append(r2)
        if r2 > opt.best_score:
            opt.best_score = r2
            opt.best_x = z.copy()
    print(f"BO for {arch_name}: pop={pop}, warm-start={len(warmstart)} points, best={opt.best_score:.4f}")
    print(f"Logging to {results_file}. Ctrl-C to stop.\n")

    gen = 0
    try:
        while True:
            zs = opt.ask(pop)
            configs = [from_probit(z, params) for z in zs]

            ids = []
            for i, cfg in enumerate(configs):
                name = f"bo_{arch_name}_g{gen}_{i}"
                script = make_script(name, cfg)
                pid = launch_pueue(script, group)
                ids.append((pid, name, cfg))
                print(f"  launched {name} (pueue {pid}): {fmt_params(cfg)}")

            wait_pueue(group)

            scores = []
            for pid, name, cfg in ids:
                r2 = parse_result(pid)
                if r2 is None:
                    print(f"  WARNING: no result for {name} (pueue {pid})")
                    r2 = 0.1
                r2 = max(r2, 0.1)
                scores.append(r2)
                log_result(results_file, cfg, r2)
                print(f"  {name}: r2={r2:.4f}")

            opt.tell(zs, np.array(scores))
            clean_pueue(group)

            best_config = from_probit(opt.best(), params)
            print(f"  gen {gen}: best_so_far={opt.best_score:.4f}  {fmt_params(best_config)}\n")
            gen += 1

    except KeyboardInterrupt:
        print(f"\nStopped after {gen} generations.")
        best_config = from_probit(opt.best(), params)
        print(f"Best: r2={opt.best_score:.4f}  {fmt_params(best_config)}")

    return opt
