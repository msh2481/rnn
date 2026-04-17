"""Shared sweep infrastructure.

CENTER: the shared baseline config.
build_model(cfg, name): constructs a TCN from a flat config dict.
Sweep: (name, make_config, values) — declarative sweep definition.
run_all(sweeps, group_name): launches all runs from all sweeps into one pueue group,
    waits once, parses results, writes per-sweep JSONL files.
"""

import base64
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import torch.optim as optim

from src.dl import LayerSpec, TCN
from src.dl.optim import MuonAdam
from src.dl.reparam import solve_scale


CENTER = {
    # architecture: 1+1 + [4]*n_blocks + 1+1, uniform ch (scaled to hit target params)
    "input_dim": 32,
    "n_blocks": 1,
    "kernel_size": 3,
    "target_log10_params": 4.9,
    "groups": 1,
    "ln_mode": "none",       # "none" | "shared" | "all"
    "nonlin_mode": "all",    # "all" | "alternating"
    "separable": False,      # depthwise + pointwise(groups=G) instead of full conv

    # optimizer: "adam" (NT-ASGD averaging) or "muonadam" (built-in averaging)
    "optimizer": "adam",
    "log10_lr": -2.5,        # Adam lr=3e-3 (opt sweep winner)
    # MuonAdam-specific (used when optimizer="muonadam")
    "log10_muon_lr": -3.0,
    "muon_momentum": 0.95,
    "log10_muon_adam_lr": -3.0,

    # regularization
    "weight_decay": 0.0,
    "dropout": 0.1,
    "input_noise": 0.0,
    "mixup_alpha": 0.0,

    # aux predictions
    "aux_horizons": [],
    "aux_weight": 0.5,

    # training
    "n_epochs": 50,
    "batch_size": 16,
    "asgd_patience": 5,
    "grad_clip": 1.0,
    "mimo": 1,

    # cross-validation (0 = single train/test holdout, >1 = k-fold over combined data)
    "cv_folds": 0,
}


def build_model(cfg: dict, name: str) -> TCN:
    n_blocks = cfg["n_blocks"]
    ks = cfg["kernel_size"]
    ln_mode = cfg["ln_mode"]
    nonlin_mode = cfg["nonlin_mode"]
    groups = cfg["groups"]

    pre_ln = (ln_mode == "all")
    shared_ln = (ln_mode in ("all", "shared"))
    sep = cfg.get("separable", False)

    specs = []
    specs.append(LayerSpec(ks, 64, n_rep=1, ln=pre_ln, nonlin=True, separable=sep))
    specs.append(LayerSpec(ks, 64, n_rep=1, ln=pre_ln, nonlin=True, separable=sep))
    for _ in range(n_blocks):
        specs.append(LayerSpec(ks, 64, n_rep=4, ln=shared_ln, nonlin=True, separable=sep))
    specs.append(LayerSpec(ks, 64, n_rep=1, ln=pre_ln, nonlin=True, separable=sep))
    specs.append(LayerSpec(ks, 64, n_rep=1, ln=pre_ln, nonlin=True, separable=sep))

    if nonlin_mode == "alternating":
        for i, s in enumerate(specs):
            s.nonlin = (i % 2 == 0)

    target = int(10 ** cfg["target_log10_params"])
    _, scaled = solve_scale(specs, target, cfg["input_dim"], groups)

    if cfg["optimizer"] == "adam":
        optimizer_fn = partial(
            optim.Adam, lr=10 ** cfg["log10_lr"],
            weight_decay=cfg["weight_decay"],
        )
    elif cfg["optimizer"] == "muonadam":
        optimizer_fn = partial(
            MuonAdam,
            lr=10 ** cfg["log10_muon_lr"],
            momentum=cfg["muon_momentum"],
            adam_lr=10 ** cfg["log10_muon_adam_lr"],
            weight_decay=cfg["weight_decay"],
            patience=cfg["asgd_patience"],
        )
    else:
        raise ValueError(f"unknown optimizer: {cfg['optimizer']}")

    aux_h = tuple(cfg["aux_horizons"])

    return TCN(
        name=name,
        input_dim=cfg["input_dim"],
        layers=scaled,
        groups=groups,
        dropout=cfg["dropout"],
        optimizer_fn=optimizer_fn,
        scheduler_fn=None,
        grad_clip=cfg["grad_clip"],
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        mimo=cfg["mimo"],
        asgd_patience=cfg["asgd_patience"],
        input_noise=cfg["input_noise"],
        aux_horizons=aux_h,
        aux_weight=cfg["aux_weight"],
        mixup_alpha=cfg["mixup_alpha"],
    )


@dataclass
class Sweep:
    name: str
    make_config: Callable[[object], dict]
    values: list


def _pueue(args: list[str]) -> str:
    return subprocess.run(["pueue"] + args, capture_output=True, text=True).stdout


def _setup_group(name: str, parallel: int | None = None):
    if parallel is None:
        parallel = int(os.environ.get("PUEUE_PARALLEL", "4"))
    _pueue(["group", "add", name])
    _pueue(["parallel", str(parallel), "--group", name])


def _launch(cfg: dict, name: str, group: str) -> int:
    payload = {**cfg, "_name": name}
    b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    out = _pueue(["add", "--group", group, "python", "-m", "sweeps.common", "--run", b64])
    m = re.search(r"id (\d+)", out)
    return int(m.group(1))


def _parse_result(pueue_id: int, metric: str) -> float | None:
    log = _pueue(["log", str(pueue_id)])
    matches = re.findall(metric + r"=([\-\d.]+)", log)
    return max(float(x) for x in matches) if matches else None


def _log_jsonl(path: Path, entry: dict):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_all(sweeps: list[Sweep], group_name: str, results_dir: str = "sweeps/results"):
    results_path = Path(results_dir)

    for sweep in sweeps:
        f = results_path / f"{sweep.name}.jsonl"
        if f.exists():
            f.unlink()

    _setup_group(group_name, parallel=4)

    launches = []
    total = sum(len(s.values) for s in sweeps)
    print(f"Launching {total} runs across {len(sweeps)} sweeps in group '{group_name}'")
    for sweep in sweeps:
        print(f"  {sweep.name}: {sweep.values}")
        for v in sweep.values:
            cfg = sweep.make_config(v)
            run_name = f"{sweep.name}_{v}"
            pid = _launch(cfg, run_name, group_name)
            launches.append((sweep.name, v, pid, cfg))

    print(f"\nWaiting for group '{group_name}' ({total} runs)...")
    subprocess.run(["pueue", "wait", "--group", group_name])

    print("\nResults:")
    for sweep_name, value, pid, cfg in launches:
        r2 = _parse_result(pid, "cv_r2")    # CV runs print cv_r2=X at end
        if r2 is None:
            r2 = _parse_result(pid, "r2_avg")
        if r2 is None:
            r2 = _parse_result(pid, "val_r2")
        entry = {"value": value, "r2": r2, "pueue_id": pid, "config": cfg}
        _log_jsonl(results_path / f"{sweep_name}.jsonl", entry)
        r2_str = f"{r2:.4f}" if r2 is not None else "FAILED"
        print(f"  {sweep_name}[{value}]: r2={r2_str}")

    _pueue(["clean", "--group", group_name])
    print(f"\nResults in {results_path}/")


def _run_from_cli():
    idx = sys.argv.index("--run")
    cfg = json.loads(base64.b64decode(sys.argv[idx + 1]).decode())
    name = cfg.pop("_name")
    print(f"config: {cfg}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    cv_folds = cfg.get("cv_folds", 0)
    if cv_folds <= 1:
        build_model(cfg, name).to(device).fit()
        return

    # CV mode: combine train+test, k-fold by seq_ix
    import numpy as np
    import pandas as pd
    from src.utils import TRAIN_FILE, TEST_FILE

    tr = pd.read_parquet(TRAIN_FILE)
    te = pd.read_parquet(TEST_FILE)
    df = pd.concat([tr, te], ignore_index=True)
    seq_col = df.columns[0]
    seq_ids = sorted(df[seq_col].unique())
    rng = np.random.default_rng(42)
    perm = rng.permutation(seq_ids)
    fold_size = len(perm) // cv_folds

    fold_r2s = []
    for fold in range(cv_folds):
        val_ids = perm[fold * fold_size : (fold + 1) * fold_size] if fold < cv_folds - 1 \
            else perm[fold * fold_size :]
        train_df = df[~df[seq_col].isin(val_ids)].reset_index(drop=True)
        val_df = df[df[seq_col].isin(val_ids)].reset_index(drop=True)
        print(f"\n=== FOLD {fold+1}/{cv_folds}: train={train_df[seq_col].nunique()} seqs, val={val_df[seq_col].nunique()} seqs ===")
        m = build_model(cfg, f"{name}_f{fold}").to(device)
        r2 = m.fit(dataset=train_df, val_dataset=val_df)
        print(f"=== FOLD {fold+1} best_r2={r2:.6f} ===")
        fold_r2s.append(r2)

    mean = float(np.mean(fold_r2s))
    std = float(np.std(fold_r2s))
    print(f"\nCV_FOLDS={cv_folds} folds={fold_r2s}")
    print(f"cv_r2={mean:.6f}  cv_std={std:.6f}")


if __name__ == "__main__":
    if "--run" in sys.argv:
        _run_from_cli()
