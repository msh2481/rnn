"""Wrapper around skopt.Optimizer with probit coordinate transform.

Supports EIps (expected improvement per unit time) — the optimizer
learns to prefer configs that are both good AND fast to evaluate.
"""

import numpy as np
from scipy import stats
from skopt import Optimizer
from skopt.space import Real

from src.optim.params import Param


class SkoptBO:
    def __init__(
        self,
        params: list[Param],
        x0: dict,
        acq_func: str = "EIps",
        n_initial: int = 10,
        seed: int | None = None,
    ):
        self.params = params
        self.dim = len(params)

        # skopt works in probit space: all dimensions are Real(-4, 4)
        # (±4 sigma covers 99.99% of the prior)
        self.space = [Real(-4.0, 4.0, name=p.name) for p in params]
        self.opt = Optimizer(
            self.space,
            base_estimator="GP",
            acq_func=acq_func,
            n_initial_points=n_initial,
            random_state=seed,
        )

        self.best_score = -np.inf
        self.best_x: dict | None = None
        self.generation = 0

        # convert x0 to probit
        self._x0_probit = [p.to_probit(x0[p.name]) for p in params]

    def _to_probit(self, config: dict) -> list[float]:
        return [float(np.clip(p.to_probit(config[p.name]), -3.99, 3.99)) for p in self.params]

    def _from_probit(self, z: list[float]) -> dict:
        return {p.name: p.from_probit(float(z[i])) for i, p in enumerate(self.params)}

    def ask(self, n: int) -> list[dict]:
        points = self.opt.ask(n)
        return [self._from_probit(z) for z in points]

    def tell(self, configs: list[dict], scores: list[float], times: list[float] | None = None):
        """Report results. If acq_func is EIps, pass times (seconds per eval)."""
        zs = [self._to_probit(c) for c in configs]
        if times is not None:
            # EIps: y must be (score, time) tuple
            ys = [(-s, t) for s, t in zip(scores, times)]
        else:
            ys = [-s for s in scores]
        self.opt.tell(zs, ys)

        for c, s in zip(configs, scores):
            if s > self.best_score:
                self.best_score = s
                self.best_x = c
        self.generation += 1

    def best(self) -> dict:
        return self.best_x if self.best_x is not None else self._from_probit(self._x0_probit)

    def warmstart(self, configs: list[dict], scores: list[float]):
        """Add prior observations."""
        zs = [self._to_probit(c) for c in configs]
        self.opt.tell(zs, [-s for s in scores])
        for c, s in zip(configs, scores):
            if s > self.best_score:
                self.best_score = s
                self.best_x = c
