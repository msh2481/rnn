import numpy as np

try:
    from ._structured_base import evaluate_model, StructuredESNBase
except ImportError:
    from _structured_base import evaluate_model, StructuredESNBase


class CubeESN(StructuredESNBase):
    def __init__(
        self,
        d: int = 4,
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
        self.a0 = None
        self.a1 = None
        self.a2 = None
        super().__init__(
            reservoir_size=d * d * d,
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
        return "CubeESN"

    def _variant_repr_fields(self):
        return [("d", self.d)]

    def _build_recurrent_operator(self):
        scale = 1.0 / np.sqrt(self.d)
        self.a0 = self.rng.normal(loc=0.0, scale=scale, size=(self.d, self.d))
        self.a1 = self.rng.normal(loc=0.0, scale=scale, size=(self.d, self.d))
        self.a2 = self.rng.normal(loc=0.0, scale=scale, size=(self.d, self.d))

    def _apply_recurrent_unscaled(self, state: np.ndarray) -> np.ndarray:
        tensor = state.reshape(self.d, self.d, self.d)
        out = np.einsum(
            "ia,jb,kc,abc->ijk",
            self.a0,
            self.a1,
            self.a2,
            tensor,
            optimize=True,
        )
        return out.reshape(-1)

    def _apply_recurrent_transpose_unscaled(self, state: np.ndarray) -> np.ndarray:
        tensor = state.reshape(self.d, self.d, self.d)
        out = np.einsum(
            "ai,bj,ck,abc->ijk",
            self.a0,
            self.a1,
            self.a2,
            tensor,
            optimize=True,
        )
        return out.reshape(-1)


if __name__ == "__main__":
    evaluate_model(CubeESN(), "cube ESN")
