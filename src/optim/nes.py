import numpy as np


class NES:
    """Natural Evolution Strategy with antithetic sampling.

    ask() returns mirrored pairs: [x+eps_1, x-eps_1, x+eps_2, x-eps_2, ...].
    tell() estimates gradient from f(x+eps) - f(x-eps), so no baseline needed.
    """

    def __init__(self, x0: np.ndarray, sigma: float, lr: float, seed: int | None = None):
        self.mean = x0.copy()
        self.sigma = sigma
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self._epsilons: np.ndarray | None = None
        self.best_score = -np.inf
        self.best_x: np.ndarray | None = None
        self.generation = 0

    def ask(self, n: int) -> np.ndarray:
        assert n % 2 == 0, "n must be even (antithetic pairs)"
        half = n // 2
        eps = self.rng.standard_normal((half, len(self.mean)))
        self._epsilons = eps
        pos = self.mean + self.sigma * eps
        neg = self.mean - self.sigma * eps
        # interleave: [+eps_0, -eps_0, +eps_1, -eps_1, ...]
        out = np.empty((n, len(self.mean)))
        out[0::2] = pos
        out[1::2] = neg
        return out

    def tell(self, xs: np.ndarray, scores: np.ndarray):
        scores = np.asarray(scores)
        best_idx = np.argmax(scores)
        if scores[best_idx] > self.best_score:
            self.best_score = scores[best_idx]
            self.best_x = xs[best_idx].copy()

        # f(x+eps) - f(x-eps) for each pair
        score_plus = scores[0::2]
        score_minus = scores[1::2]
        diff = score_plus - score_minus  # (half,)

        # divide by 2*sigma to get actual gradient estimate
        # f(x+σε) - f(x-σε) ≈ 2σ * ε·∇f, so ∇f ≈ ε * diff / (2σ)
        grad = (self._epsilons.T @ diff) / (len(diff) * 2 * self.sigma)
        self.mean += self.lr * grad
        self._epsilons = None
        self.generation += 1

    def best(self) -> np.ndarray:
        return self.best_x if self.best_x is not None else self.mean.copy()
