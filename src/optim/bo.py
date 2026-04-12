import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


class BO:
    """Bayesian Optimization with GP surrogate (quadratic kernel).

    First few rounds: random samples near x0.
    Then: fit GP to all observations, sample many candidates, pick top-n by UCB.
    """

    def __init__(
        self,
        x0: np.ndarray,
        sigma: float,
        bounds: tuple[float, float] = (-2.0, 2.0),
        ucb_kappa: float = 1.0,
        n_candidates: int = 1000,
        seed: int | None = None,
    ):
        self.x0 = x0.copy()
        self.dim = len(x0)
        self.sigma = sigma
        self.bounds = bounds
        self.ucb_kappa = ucb_kappa
        self.n_candidates = n_candidates
        self.rng = np.random.default_rng(seed)

        self.obs_x: list[np.ndarray] = []
        self.obs_y: list[float] = []
        self.best_score = -np.inf
        self.best_x: np.ndarray | None = None
        self.generation = 0

        # DotProduct^2 gives a quadratic kernel: k(x,y) = (sigma_0^2 + x·y)^2
        # plus WhiteKernel for noise
        kernel = DotProduct() ** 2 + WhiteKernel()
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=seed)

    def ask(self, n: int) -> np.ndarray:
        if len(self.obs_x) < 2 * self.dim:
            # not enough data to fit GP — random exploration
            return self.x0 + self.sigma * self.rng.standard_normal((n, self.dim))

        X = np.array(self.obs_x)
        y = np.array(self.obs_y)
        self.gp.fit(X, y)

        candidates = self.rng.uniform(
            self.bounds[0], self.bounds[1], size=(self.n_candidates, self.dim)
        )
        mu, std = self.gp.predict(candidates, return_std=True)
        ucb = mu + self.ucb_kappa * std

        top_idx = np.argsort(-ucb)[:n]
        return candidates[top_idx]

    def tell(self, xs: np.ndarray, scores: np.ndarray):
        scores = np.asarray(scores)
        for x, s in zip(xs, scores):
            self.obs_x.append(x.copy())
            self.obs_y.append(s)
            if s > self.best_score:
                self.best_score = s
                self.best_x = x.copy()
        self.generation += 1

    def best(self) -> np.ndarray:
        return self.best_x if self.best_x is not None else self.x0.copy()
