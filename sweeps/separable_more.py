"""3 more separable conv variants at CV=0 (single-holdout for speed)."""

from sweeps.common import Sweep, CENTER, run_all


SEP = {**CENTER, "separable": True, "cv_folds": 0}


sweeps = [
    # With groups=2 (another way to split channels, might interact with separable)
    Sweep("sep_g2", lambda _: {**SEP, "groups": 2}, values=[0]),
    # 2x capacity — separable has ~3x fewer params, so we can go much wider
    Sweep("sep_wide", lambda _: {**SEP, "target_log10_params": 5.2}, values=[0]),
    # With wd + ln (the winning pair from stage 1)
    Sweep("sep_wdln", lambda _: {**SEP, "weight_decay": 1e-4, "ln_mode": "shared"}, values=[0]),
]


if __name__ == "__main__":
    run_all(sweeps, group_name="sw_sep_more")
