import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


class BO:
    """Bayesian Optimization with GP surrogate (quadratic kernel).

    ask(n) returns n points:
      - First point: argmax of GP mean over random candidates (exploitation).
      - Remaining n-1 points: sampled from a Gaussian fitted to the top 50%
        of observed points (CMA-ES-style exploration).

    Before enough data is collected, returns random samples near x0.
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

        kernel = DotProduct() ** 2 + WhiteKernel()
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=seed)

    def _gp_optimum(self) -> np.ndarray:
        candidates = self.rng.uniform(
            self.bounds[0], self.bounds[1], size=(self.n_candidates, self.dim)
        )
        mu, std = self.gp.predict(candidates, return_std=True)
        ucb = mu + self.ucb_kappa * std
        return candidates[np.argmax(ucb)]

    def _elite_sample(self, n: int) -> np.ndarray:
        X = np.array(self.obs_x)
        y = np.array(self.obs_y)
        # top 50%
        k = max(2, len(y) // 2)
        top_idx = np.argsort(-y)[:k]
        elite = X[top_idx]
        mu = elite.mean(axis=0)
        diff = elite - mu
        cov = (diff.T @ diff) / len(elite) + 1e-4 * np.eye(self.dim)
        return self.rng.multivariate_normal(mu, cov, size=n)

    def ask(self, n: int) -> np.ndarray:
        if len(self.obs_x) < 2 * self.dim:
            return self.x0 + self.sigma * self.rng.standard_normal((n, self.dim))

        X = np.array(self.obs_x)
        y = np.array(self.obs_y)
        self.gp.fit(X, y)

        out = np.empty((n, self.dim))
        out[0] = self._gp_optimum()
        if n > 1:
            out[1:] = self._elite_sample(n - 1)
        return out

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
