"""Stage 1: pair and triple combinations of per-axis winners, under CV=4."""

from sweeps.common import Sweep, CENTER, run_all


# All runs use CV=4
CV = {**CENTER, "cv_folds": 4}


def combo(**overrides) -> dict:
    return {**CV, **overrides}


sweeps = [
    # Baseline (CV=4 center)
    Sweep("center", lambda _: combo(), values=[0]),

    # Pairs with groups=2 (strongest single lever)
    Sweep("g2_wd", lambda _: combo(groups=2, weight_decay=1e-4), values=[0]),
    Sweep("g2_nonlin", lambda _: combo(groups=2, nonlin_mode="alternating"), values=[0]),
    Sweep("g2_ln", lambda _: combo(groups=2, ln_mode="shared"), values=[0]),
    Sweep("g2_aux", lambda _: combo(groups=2, aux_horizons=[8]), values=[0]),

    # Pairs without groups
    Sweep("wd_nonlin", lambda _: combo(weight_decay=1e-4, nonlin_mode="alternating"), values=[0]),
    Sweep("wd_ln", lambda _: combo(weight_decay=1e-4, ln_mode="shared"), values=[0]),
    Sweep("ln_nonlin", lambda _: combo(ln_mode="shared", nonlin_mode="alternating"), values=[0]),
    Sweep("k5_ln", lambda _: combo(kernel_size=5, ln_mode="shared"), values=[0]),

    # Triples with the strongest pair (g2 + wd) plus one more
    Sweep("g2_wd_nonlin", lambda _: combo(groups=2, weight_decay=1e-4, nonlin_mode="alternating"), values=[0]),
    Sweep("g2_wd_ln", lambda _: combo(groups=2, weight_decay=1e-4, ln_mode="shared"), values=[0]),
]


if __name__ == "__main__":
    run_all(sweeps, group_name="sw_pairs")
