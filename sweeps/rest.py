"""Wave 2: architecture, regularization, training tricks, MuonAdam optimizer search."""

from sweeps.common import Sweep, CENTER, run_all


# MuonAdam center: optimizer="muonadam", default muon_lr=1e-3, momentum=0.95
def muon_base() -> dict:
    return {**CENTER, "optimizer": "muonadam"}


sweeps = [
    # Architecture — center values excluded (n_blocks=1, ks=3, groups=1, log_params=4.9)
    Sweep("n_blocks", lambda n: {**CENTER, "n_blocks": n},
          values=[0, 2, 3, 4]),
    Sweep("kernel_size", lambda k: {**CENTER, "kernel_size": k},
          values=[4, 5, 8]),
    Sweep("groups", lambda g: {**CENTER, "groups": g},
          values=[2, 4, 8]),
    Sweep("log_params", lambda lp: {**CENTER, "target_log10_params": lp},
          values=[4.0, 4.3, 4.6, 5.2, 5.5]),

    # Regularization — per-axis since center has wd=0, no geomean constraint
    Sweep("dropout", lambda v: {**CENTER, "dropout": v},
          values=[0.0, 0.03, 0.3]),
    Sweep("weight_decay", lambda v: {**CENTER, "weight_decay": v},
          values=[1e-4, 1e-3, 1e-2, 1e-1]),
    Sweep("input_noise", lambda v: {**CENTER, "input_noise": v},
          values=[0.03, 0.1, 0.3]),
    Sweep("mixup", lambda v: {**CENTER, "mixup_alpha": v},
          values=[0.05, 0.1, 0.3]),

    # Structure tricks
    Sweep("ln", lambda m: {**CENTER, "ln_mode": m}, values=["shared"]),
    Sweep("nonlin", lambda m: {**CENTER, "nonlin_mode": m}, values=["alternating"]),

    # Aux predictions
    Sweep("aux_k", lambda k: {**CENTER, "aux_horizons": [k]},
          values=[2, 4, 8]),

    # MuonAdam optimizer search (treats muonadam center as an additional data point)
    Sweep("muon_center", lambda _: muon_base(), values=[0]),
    Sweep("muon_lr", lambda v: {**muon_base(), "log10_muon_lr": v},
          values=[-4.0, -3.5, -2.5, -2.0]),       # center (-3.0) covered by muon_center
    Sweep("muon_momentum", lambda v: {**muon_base(), "muon_momentum": v},
          values=[0.85, 0.90, 0.97, 0.99]),        # center (0.95) covered by muon_center
]

if __name__ == "__main__":
    run_all(sweeps, group_name="sw_rest")
