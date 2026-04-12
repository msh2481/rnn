import numpy as np
from scipy.stats import special_ortho_group

from src.esn.esn import ESN
from src.utils import DataPoint


class ES2N(ESN):
    def __init__(
        self,
        reservoir_size: int = 256,
        beta=None,
        nonlinearity=None,
        spectral_radius=None,
        leak_rate=None,
        input_scale=None,
        bias_scale=None,
        density=None,
        ridge_alpha=None,
        seed=None,
    ):
        super().__init__(
            reservoir_size=reservoir_size,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            input_scale=input_scale,
            bias_scale=bias_scale,
            density=density,
            nonlinearity=nonlinearity,
            ridge_alpha=ridge_alpha,
            seed=seed,
        )

        self.beta = beta
        if self.beta is None:
            self.beta = self.rng.uniform(0.0, 0.5)
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
            f"leak_rate={self.leak_rate}, "
            f"input_scale={self.input_scale}, "
            f"bias_scale={self.bias_scale}, "
            f"density={self.density}, "
            f"ridge_alpha={self.ridge_alpha})"
        )

    def _ensure_initialized(self, input_dim: int):
        if (
            self.input_dim == input_dim
            and self.w_in is not None
            and self.orthogonal is not None
        ):
            return

        self.orthogonal = None
        super()._ensure_initialized(input_dim)
        if self.orthogonal is None:
            self.orthogonal = special_ortho_group.rvs(
                self.reservoir_size, random_state=self.rng
            )

    def _advance_state(self, data_point: DataPoint):
        self._ensure_initialized(data_point.state.shape[0])

        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence(data_point.seq_ix)

        pre_activation = (
            self.w_res @ self.state + self.w_in @ data_point.state + self.bias
        )
        nonlinear_state = self._activate(pre_activation)
        linear_state = self.orthogonal @ self.state
        esn_state = (1 - self.leak_rate) * self.state + self.leak_rate * nonlinear_state
        self.state = (1 - self.beta) * esn_state + self.beta * linear_state


if __name__ == "__main__":
    from src.utils import train_and_eval
    train_and_eval(ES2N())
