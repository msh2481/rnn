import numpy as np

try:
    from ._structured_base import StructuredESNBase, evaluate_model
except ImportError:
    from _structured_base import StructuredESNBase, evaluate_model


class SquareESN(StructuredESNBase):
    def __init__(
        self,
        d: int = 64,
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
        self.a_left = None
        self.a_right = None
        super().__init__(
            reservoir_size=d * d,
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
        return "SquareESN"

    def _variant_repr_fields(self):
        return [("d", self.d)]

    def _build_recurrent_operator(self):
        scale = 1.0 / np.sqrt(self.d)
        self.a_left = self.rng.normal(loc=0.0, scale=scale, size=(self.d, self.d))
        self.a_right = self.rng.normal(loc=0.0, scale=scale, size=(self.d, self.d))

    def _apply_recurrent_unscaled(self, state: np.ndarray) -> np.ndarray:
        matrix = state.reshape(self.d, self.d)
        return (self.a_left @ matrix @ self.a_right.T).reshape(-1)

    def _apply_recurrent_transpose_unscaled(self, state: np.ndarray) -> np.ndarray:
        matrix = state.reshape(self.d, self.d)
        return (self.a_left.T @ matrix @ self.a_right).reshape(-1)


if __name__ == "__main__":
    evaluate_model(SquareESN(), "square ESN")
