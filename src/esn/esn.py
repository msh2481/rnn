import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.utils import DataPoint


class ESN:
    def __init__(
        self,
        reservoir_size=None,
        spectral_radius=None,
        leak_rate=None,
        input_scale=None,
        bias_scale=None,
        density=None,
        nonlinearity=None,
        ridge_alpha=None,
        seed=None,
    ):
        self.rng = np.random.default_rng(seed)
        if nonlinearity is None:
            nonlinearity = "tanh"
        self.nonlinearity = nonlinearity.lower()
        if self.nonlinearity not in {"relu", "gelu", "tanh", "hardtanh"}:
            raise ValueError(
                "nonlinearity must be one of: 'relu', 'gelu', 'tanh', 'hardtanh'"
            )

        self.reservoir_size = reservoir_size
        if self.reservoir_size is None:
            self.reservoir_size = 1024

        self.spectral_radius = spectral_radius
        if self.spectral_radius is None:
            self.spectral_radius = self.rng.uniform(0.8, 1.0)

        self.leak_rate = leak_rate
        if self.leak_rate is None:
            self.leak_rate = np.exp(self.rng.uniform(np.log(0.1), np.log(0.2)))

        self.input_scale = input_scale
        if self.input_scale is None:
            self.input_scale = 0.4

        self.bias_scale = bias_scale
        if self.bias_scale is None:
            self.bias_scale = 0.05

        self.density = density
        if self.density is None:
            self.density = self.rng.uniform(0.08, 0.24)

        self.ridge_alpha = ridge_alpha
        if self.ridge_alpha is None:
            self.ridge_alpha = np.exp(self.rng.uniform(np.log(0.01), np.log(0.1)))

        self.current_seq_ix = None
        self.state = None
        self.input_dim = None
        self.w_in = None
        self.w_res = None
        self.bias = None
        self.w_out = None

    def __repr__(self):
        return (
            "ESN("
            f"reservoir_size={self.reservoir_size}, "
            f"spectral_radius={self.spectral_radius}, "
            f"leak_rate={self.leak_rate}, "
            f"input_scale={self.input_scale}, "
            f"bias_scale={self.bias_scale}, "
            f"density={self.density}, "
            f"nonlinearity='{self.nonlinearity}', "
            f"ridge_alpha={self.ridge_alpha})"
        )

    def _ensure_initialized(self, input_dim: int):
        if self.input_dim == input_dim and self.w_in is not None:
            return

        self.input_dim = input_dim
        self.state = np.zeros(self.reservoir_size)

        self.w_in = self.rng.normal(
            loc=0.0,
            scale=self.input_scale / np.sqrt(max(1, input_dim)),
            size=(self.reservoir_size, input_dim),
        )

        # Sparse random reservoir rescaled to the target spectral radius.
        mask = (
            self.rng.random((self.reservoir_size, self.reservoir_size)) < self.density
        )
        raw_res = self.rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(1, self.reservoir_size * self.density)),
            size=(self.reservoir_size, self.reservoir_size),
        )
        self.w_res = raw_res * mask

        current_radius = self._spectral_radius(self.w_res)
        if current_radius > 0:
            self.w_res *= self.spectral_radius / current_radius

        self.bias = self.rng.normal(
            loc=0.0, scale=self.bias_scale, size=self.reservoir_size
        )

    def _spectral_radius(self, matrix: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvals(matrix)
        return float(np.max(np.abs(eigenvalues)))

    def _reset_sequence(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.state = np.zeros(self.reservoir_size)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.nonlinearity == "relu":
            return np.maximum(x, 0.0)
        if self.nonlinearity == "gelu":
            return (
                0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
            )
        if self.nonlinearity == "hardtanh":
            return np.clip(x, -1.0, 1.0)
        return np.tanh(x)

    def _advance_state(self, data_point: DataPoint):
        self._ensure_initialized(data_point.state.shape[0])

        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence(data_point.seq_ix)

        pre_activation = (
            self.w_in @ data_point.state + self.w_res @ self.state + self.bias
        )
        candidate_state = self._activate(pre_activation)
        self.state = (
            1 - self.leak_rate
        ) * self.state + self.leak_rate * candidate_state

    def _readout_features(self, state_vector: np.ndarray) -> np.ndarray:
        return np.concatenate([state_vector, self.state, np.array([1.0])])

    def fit(self, dataset, show_progress: bool = True):
        if isinstance(dataset, str):
            dataset = pd.read_parquet(dataset)
        rows = tqdm(dataset.values) if show_progress else dataset.values

        gram = None
        rhs = None
        next_features = None

        self.current_seq_ix = None
        self.state = None

        for row in rows:
            seq_ix = row[0]
            step_in_seq = row[1]
            need_prediction = row[2]
            new_state = row[3:]

            if next_features is not None:
                if gram is None:
                    feature_dim = next_features.shape[0]
                    output_dim = new_state.shape[0]
                    gram = np.eye(feature_dim) * self.ridge_alpha
                    rhs = np.zeros((feature_dim, output_dim))
                gram += np.outer(next_features, next_features)
                rhs += np.outer(next_features, new_state)

            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, new_state)
            self._advance_state(data_point)

            if need_prediction:
                next_features = self._readout_features(new_state)
            else:
                next_features = None

        if gram is None or rhs is None:
            raise ValueError("No training examples were collected for the ESN readout.")

        self.w_out = np.linalg.solve(gram, rhs)
        self.current_seq_ix = None
        self.state = np.zeros(self.reservoir_size)
        return self

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.w_out is None:
            raise ValueError(
                "ESN readout is not trained. Call .fit(dataset_path) first."
            )

        self._advance_state(data_point)
        features = self._readout_features(data_point.state)
        return features @ self.w_out


if __name__ == "__main__":
    from src.utils import train_and_eval
    train_and_eval(ESN())
