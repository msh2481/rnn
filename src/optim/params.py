"""Parameter specifications with priors for BO in probit space.

Each param has a scipy.stats distribution as its prior.
Probit coordinate: z = Φ^{-1}(CDF(actual_value))
Inverse: actual_value = PPF(Φ(z))
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Param:
    name: str
    prior: stats.rv_continuous  # frozen scipy distribution

    def to_probit(self, value: float) -> float:
        return stats.norm.ppf(np.clip(self.prior.cdf(value), 1e-6, 1 - 1e-6))

    def from_probit(self, z: float) -> float:
        return self.prior.ppf(np.clip(stats.norm.cdf(z), 1e-6, 1 - 1e-6))


def preview_priors(params: list[Param], n=10000):
    rng = np.random.default_rng(0)
    quantiles = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    header = "param           " + "".join(f"{'q'+str(int(q*100)):>8s}" for q in quantiles)
    print(header)
    print("-" * len(header))
    for p in params:
        samples = p.prior.rvs(n, random_state=rng)
        vals = np.quantile(samples, quantiles)
        row = f"{p.name:16s}" + "".join(f"{v:8.5f}" for v in vals)
        print(row)
