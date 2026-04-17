"""Thompson sampling BO via Bayesian linear regression in RKHS.

The DotProduct()**2 kernel k(x,x') = (x·x' + σ₀²)² has a finite-dimensional
feature map (degree-2 polynomial features), so the GP posterior is exactly a
Bayesian linear regression.  Each ask() draws weight vectors from the BLR
posterior, then finds the analytic optimum of each sampled quadratic.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


class SampleBO:

    def __init__(
        self,
        x0: np.ndarray,
        sigma: float,
        bounds: tuple[float, float] = (-2.0, 2.0),
        temperature: float = 1.0,
        n_candidates: int = 1000,
        seed: int | None = None,
    ):
        self.x0 = x0.copy()
        self.dim = len(x0)
        self.sigma = sigma
        self.bounds = bounds
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.rng = np.random.default_rng(seed)

        self.obs_x: list[np.ndarray] = []
        self.obs_y: list[float] = []
        self.best_score = -np.inf
        self.best_x: np.ndarray | None = None
        self.generation = 0

        kernel = DotProduct() ** 2 + WhiteKernel()
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2, random_state=seed,
        )

    # -- feature map ----------------------------------------------------------

    def _features(self, X: np.ndarray, sigma_0: float) -> np.ndarray:
        """Degree-2 polynomial features for k(x,x') = (x·x' + σ₀²)².

        Layout (D = 1 + d(d+3)/2 columns):
          [σ₀²,  √2·σ₀·x₁ … √2·σ₀·x_d,  x₁² … x_d²,  √2·x₁x₂ … ]
        """
        n, d = X.shape
        parts = [
            np.full((n, 1), sigma_0 ** 2),
            np.sqrt(2) * sigma_0 * X,
            X ** 2,
        ]
        for i in range(d):
            for j in range(i + 1, d):
                parts.append((np.sqrt(2) * X[:, i] * X[:, j])[:, None])
        return np.hstack(parts)

    # -- analytic quadratic optimum -------------------------------------------

    def _quadratic_optimum(self, w: np.ndarray, sigma_0: float):
        """Analytic optimum of f(x) = w · φ(x).  Returns None if not concave."""
        d = self.dim
        # f(x) = w₀σ₀² + Σ w_lin·√2σ₀·xᵢ + Σ w_sq·xᵢ² + Σ w_cr·√2·xᵢxⱼ
        #       = const  + aᵀx              + ½ xᵀHx
        a = np.sqrt(2) * sigma_0 * w[1 : d + 1]

        H = np.diag(2.0 * w[d + 1 : 2 * d + 1])
        idx = 2 * d + 1
        for i in range(d):
            for j in range(i + 1, d):
                H[i, j] = H[j, i] = np.sqrt(2) * w[idx]
                idx += 1

        eigvals = np.linalg.eigvalsh(H)
        if not np.all(eigvals < -1e-8):
            return None
        x_star = np.linalg.solve(H, -a)
        return np.clip(x_star, *self.bounds)

    def _fallback_optimum(self, w: np.ndarray, sigma_0: float) -> np.ndarray:
        """Argmax of sampled quadratic over random candidates."""
        cands = self.rng.uniform(*self.bounds, size=(self.n_candidates, self.dim))
        return cands[np.argmax(self._features(cands, sigma_0) @ w)]

    # -- public interface (matches BO) ----------------------------------------

    def ask(self, n: int) -> np.ndarray:
        if len(self.obs_x) < 2 * self.dim:
            return self.x0 + self.sigma * self.rng.standard_normal((n, self.dim))

        X = np.array(self.obs_x)
        y = np.array(self.obs_y)
        self.gp.fit(X, y)

        # fitted hyperparameters
        kern = self.gp.kernel_
        sigma_0 = kern.k1.kernel.sigma_0          # DotProduct inside Exponentiation
        noise = kern.k2.noise_level                # WhiteKernel

        # BLR posterior  w | data ~ N(m, S)
        Phi = self._features(X, sigma_0)
        D = Phi.shape[1]
        A = Phi.T @ Phi / noise + np.eye(D)
        L, low = cho_factor(A)
        m = cho_solve((L, low), Phi.T @ y / noise)
        # Cholesky of S = A⁻¹  for sampling
        S_chol = np.linalg.cholesky(cho_solve((L, low), np.eye(D))
                                     + 1e-10 * np.eye(D))
        S_chol *= self.temperature

        out = np.empty((n, self.dim))
        for i in range(n):
            w = m + S_chol @ self.rng.standard_normal(D)
            opt = self._quadratic_optimum(w, sigma_0)
            if opt is None:
                opt = self._fallback_optimum(w, sigma_0)
            out[i] = opt
        return out

    def tell(self, xs: np.ndarray, scores: np.ndarray):
        scores = np.asarray(scores)
        for x, s in zip(xs, scores):
            self.obs_x.append(x.copy())
            self.obs_y.append(float(s))
            if s > self.best_score:
                self.best_score = s
                self.best_x = x.copy()
        self.generation += 1

    def best(self) -> np.ndarray:
        return self.best_x if self.best_x is not None else self.x0.copy()
