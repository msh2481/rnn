"""Wave 3: combined winners + geomean regularization sweep.

All 8 per-axis winners combined; then scale the 4 regularization params
(dropout, weight_decay, input_noise, mixup_alpha) jointly by 10^log_k.
"""

from sweeps.common import Sweep, CENTER, run_all


COMBINED = {
    **CENTER,
    "kernel_size": 5,
    "ln_mode": "shared",
    "nonlin_mode": "alternating",
    "aux_horizons": [8],
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "input_noise": 0.3,
    "mixup_alpha": 0.05,
}


def geomean_cfg(log_k: float) -> dict:
    k = 10 ** log_k
    return {
        **COMBINED,
        "dropout": min(0.95, 0.3 * k),
        "weight_decay": 1e-4 * k,
        "input_noise": 0.3 * k,
        "mixup_alpha": min(2.0, 0.05 * k),
    }


sweeps = [
    # log_k=0 is the full combined config; span reduced (left) to amplified (right)
    Sweep("combined_geomean", geomean_cfg,
          values=[-1.5, -1.0, -0.5, 0.0, 0.3, 0.5, 1.0]),
]

if __name__ == "__main__":
    run_all(sweeps, group_name="sw_combined")
