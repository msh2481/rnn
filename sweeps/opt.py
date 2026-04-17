"""Wave 1: optimizer sweeps (Adam lr)."""

from sweeps.common import Sweep, CENTER, run_all

sweeps = [
    # Center at log10_lr=-3.0 (lr=1e-3)
    Sweep("lr", lambda v: {**CENTER, "log10_lr": v},
          values=[-4.0, -3.5, -3.0, -2.5, -2.0]),
]

if __name__ == "__main__":
    run_all(sweeps, group_name="sw_opt")
