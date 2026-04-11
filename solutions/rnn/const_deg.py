import numpy as np

try:
    from ._structured_base import StructuredESNBase, evaluate_model
except ImportError:
    from _structured_base import StructuredESNBase, evaluate_model


class ConstDegESN(StructuredESNBase):
    def __init__(
        self,
        d: int = 1024,
        deg: int = 10,
        spectral_radius=None,
        leak_rate=None,
        input_scale=None,
        bias_scale=None,
        optimizer=None,
        lr=None,
        weight_decay=None,
        batch_size=None,
        seed=None,
    ):
        self.d = d
        self.deg = deg
        self.sources = None
        self.weights = None
        super().__init__(
            reservoir_size=d,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            input_scale=input_scale,
            bias_scale=bias_scale,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed,
        )

    @property
    def model_name(self) -> str:
        return "ConstDegESN"

    def _variant_repr_fields(self):
        return [("d", self.d), ("deg", self.deg)]

    def _build_recurrent_operator(self):
        self.sources = self.rng.integers(0, self.d, size=(self.d, self.deg))
        self.weights = self.rng.normal(
            loc=0.0, scale=1.0 / np.sqrt(max(1, self.deg)), size=(self.d, self.deg)
        )

    def _apply_recurrent_unscaled(self, state: np.ndarray) -> np.ndarray:
        return np.sum(self.weights * state[self.sources], axis=1)

    def _apply_recurrent_transpose_unscaled(self, state: np.ndarray) -> np.ndarray:
        result = np.zeros(self.d)
        for edge_ix in range(self.deg):
            np.add.at(
                result,
                self.sources[:, edge_ix],
                self.weights[:, edge_ix] * state,
            )
        return result


if __name__ == "__main__":
    evaluate_model(ConstDegESN(), "constant-degree ESN")
