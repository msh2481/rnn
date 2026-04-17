"""Launch the g2+wd+ln triple at CV=4, and separable conv at CV=0."""

from sweeps.common import Sweep, CENTER, run_all


sweeps = [
    # Triple (confirm if wd_ln + groups=2 stacks further). CV=4 (expensive).
    Sweep("g2_wd_ln_cv4", lambda _: {
        **CENTER,
        "cv_folds": 4,
        "groups": 2,
        "weight_decay": 1e-4,
        "ln_mode": "shared",
    }, values=[0]),

    # Separable conv at CV=0 (fast, just exploring).
    Sweep("separable_cv0", lambda _: {
        **CENTER,
        "cv_folds": 0,
        "separable": True,
    }, values=[0]),
]


if __name__ == "__main__":
    run_all(sweeps, group_name="sw_triple_sep")
