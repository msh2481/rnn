"""Reparametrization utilities for hyperparameter search.

Architecture scaling: count_params, solve_scale, scale_layers
Regularization: normalize_geomean
Optimizer: eff_lr_to_params, params_to_eff_lr
"""

import numpy as np

from src.dl.tcn import LayerSpec


def round_channels(base: float, groups: int) -> int:
    return max(groups, round(base / groups) * groups)


def scale_layers(layers: list[LayerSpec], k: float, groups: int) -> list[LayerSpec]:
    return [
        LayerSpec(
            kernel_size=s.kernel_size,
            channels=round_channels(s.channels * k, groups),
            n_rep=s.n_rep,
            ln=s.ln,
            nonlin=s.nonlin,
            dropout_mul=s.dropout_mul,
            separable=s.separable,
        )
        for s in layers
    ]


def count_params(layers: list[LayerSpec], input_dim: int, groups: int) -> int:
    total = 0
    chs = [s.channels for s in layers]
    first_ch, last_ch = chs[0], chs[-1]

    # proj_in: Conv1d(input_dim, first_ch, 1)
    total += input_dim * first_ch + first_ch

    prev_ch = first_ch
    for spec in layers:
        ch = spec.channels
        if prev_ch != ch:
            total += prev_ch * ch + ch
        if spec.separable:
            # depthwise Conv1d(ch, ch, ks, groups=ch): ch*(ch//ch)*ks + ch = ch*ks + ch
            total += ch * spec.kernel_size + ch
            # pointwise Conv1d(ch, ch, 1, groups=groups): ch*(ch//groups) + ch
            total += ch * (ch // groups) + ch
        else:
            # Conv1d(ch, ch, ks, groups=groups)
            total += ch * (ch // groups) * spec.kernel_size + ch
        if spec.ln:
            total += 2 * ch
        prev_ch = ch

    # readout: Conv1d(last_ch, input_dim * groups, 1, groups=groups)
    total += input_dim * last_ch + input_dim * groups

    return total


def solve_scale(
    base_layers: list[LayerSpec],
    target_params: int,
    input_dim: int,
    groups: int,
) -> tuple[float, list[LayerSpec]]:
    lo, hi = 0.01, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        scaled = scale_layers(base_layers, mid, groups)
        if count_params(scaled, input_dim, groups) < target_params:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    scaled = scale_layers(base_layers, k, groups)
    return k, scaled


def normalize_geomean(values: list[float], target: float) -> list[float]:
    current = np.exp(np.mean(np.log(values)))
    factor = target / current
    return [v * factor for v in values]


def eff_lr_to_params(log10_eff_lr: float, log10_lr: float) -> tuple[float, float]:
    """(log10_eff_lr, log10_lr) -> (lr, momentum).

    eff_lr = lr / (1 - momentum)
    momentum = 1 - 10^(log10_lr - log10_eff_lr)
    """
    lr = 10 ** log10_lr
    momentum = 1.0 - 10 ** (log10_lr - log10_eff_lr)
    return lr, max(0.0, min(momentum, 0.999))


def params_to_eff_lr(lr: float, momentum: float) -> tuple[float, float]:
    """(lr, momentum) -> (log10_eff_lr, log10_lr)."""
    eff_lr = lr / (1 - momentum)
    return np.log10(eff_lr), np.log10(lr)
