import os
import sys

import numpy as np
from scipy.stats import special_ortho_group

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")

from solutions.rnn.esn import ESN
from utils import DataPoint


class ES2N(ESN):
    def __init__(
        self,
        reservoir_size: int = 256,
        beta=None,
        nonlinearity=None,
        spectral_radius=None,
        input_scale=None,
        bias_scale=None,
        ridge_alpha=None,
        seed=None,
    ):
        super().__init__(
            reservoir_size=reservoir_size,
            spectral_radius=spectral_radius,
            leak_rate=1.0,
            input_scale=input_scale,
            bias_scale=0.0 if bias_scale is None else bias_scale,
            density=1.0,
            nonlinearity=nonlinearity,
            ridge_alpha=ridge_alpha,
            seed=seed,
        )

        self.beta = beta
        if self.beta is None:
            self.beta = self.rng.uniform(0.02, 0.25)
        self.orthogonal = None

    def _build_recurrent_operator(self):
        raw_res = self.rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(1, self.reservoir_size)),
            size=(self.reservoir_size, self.reservoir_size),
        )
        current_radius = self._spectral_radius(raw_res)
        if current_radius > 0:
            raw_res *= self.spectral_radius / current_radius
        self.w_res = raw_res

        self.orthogonal = special_ortho_group.rvs(
            self.reservoir_size, random_state=self.rng
        )

    def __repr__(self):
        return (
            "ES2N("
            f"reservoir_size={self.reservoir_size}, "
            f"beta={self.beta}, "
            f"nonlinearity='{self.nonlinearity}', "
            f"spectral_radius={self.spectral_radius}, "
            f"input_scale={self.input_scale}, "
            f"bias_scale={self.bias_scale}, "
            f"ridge_alpha={self.ridge_alpha})"
        )

    def _ensure_initialized(self, input_dim: int):
        if (
            self.input_dim == input_dim
            and self.w_in is not None
            and self.orthogonal is not None
        ):
            return

        self.input_dim = input_dim
        self.state = np.zeros(self.reservoir_size)

        self.w_in = self.rng.normal(
            loc=0.0,
            scale=self.input_scale / np.sqrt(max(1, input_dim)),
            size=(self.reservoir_size, input_dim),
        )
        self.bias = self.rng.normal(
            loc=0.0, scale=self.bias_scale, size=self.reservoir_size
        )
        self._build_recurrent_operator()

    def _advance_state(self, data_point: DataPoint):
        self._ensure_initialized(data_point.state.shape[0])

        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence(data_point.seq_ix)

        pre_activation = (
            self.w_res @ self.state + self.w_in @ data_point.state + self.bias
        )
        nonlinear_state = self._activate(pre_activation)
        linear_state = self.orthogonal @ self.state
        self.state = self.beta * nonlinear_state + (1 - self.beta) * linear_state


if __name__ == "__main__":
    from utils import ScorerStepByStep

    train_file = f"{CURRENT_DIR}/../../datasets/train.parquet"
    test_file = f"{CURRENT_DIR}/../../datasets/test.parquet"

    model = ES2N()
    print(model)
    print("Training ES2N readout...")
    model.train(train_file)
    scorer = ScorerStepByStep(test_file)
    print("Testing ES2N...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")
    results = scorer.score(model)
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")
    print(f"MSE score: {results['mse_score']:.6f}")
