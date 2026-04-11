import numpy as np

try:
    from ._structured_base import evaluate_model, StructuredESNBase
except ImportError:
    from _structured_base import evaluate_model, StructuredESNBase


class LoraESN(StructuredESNBase):
    def __init__(
        self,
        d: int = 256,
        r: int = 32,
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
        self.r = r
        self.diag = None
        self.u = None
        self.v = None
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
        return "LoraESN"

    def _variant_repr_fields(self):
        return [("d", self.d), ("r", self.r)]

    def _build_recurrent_operator(self):
        self.diag = self.rng.normal(loc=0.0, scale=1.0, size=self.d)
        scale = 1.0 / np.sqrt(max(1, self.r))
        self.u = self.rng.normal(loc=0.0, scale=scale, size=(self.d, self.r))
        self.v = self.rng.normal(loc=0.0, scale=scale, size=(self.d, self.r))

    def _apply_recurrent_unscaled(self, state: np.ndarray) -> np.ndarray:
        return self.diag * state + self.u @ (self.v.T @ state)

    def _apply_recurrent_transpose_unscaled(self, state: np.ndarray) -> np.ndarray:
        return self.diag * state + self.v @ (self.u.T @ state)


if __name__ == "__main__":
    evaluate_model(LoraESN(), "LoRA ESN")
