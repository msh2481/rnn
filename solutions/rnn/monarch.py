import numpy as np

try:
    from ._structured_base import evaluate_model, StructuredESNBase
except ImportError:
    from _structured_base import evaluate_model, StructuredESNBase


class MonarchESN(StructuredESNBase):
    def __init__(
        self,
        d: int = 1024,
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
        self.side = int(round(np.sqrt(d)))
        if self.side * self.side != d:
            raise ValueError("MonarchESN requires d to be a perfect square.")

        self.blocks1 = None
        self.blocks2 = None
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
        return "MonarchESN"

    def _variant_repr_fields(self):
        return [("d", self.d)]

    def _build_recurrent_operator(self):
        scale = 1.0 / np.sqrt(self.side)
        shape = (self.side, self.side, self.side)
        self.blocks1 = self.rng.normal(loc=0.0, scale=scale, size=shape)
        self.blocks2 = self.rng.normal(loc=0.0, scale=scale, size=shape)

    def _apply_blockdiag(self, blocks: np.ndarray, state: np.ndarray, transpose: bool):
        matrix = state.reshape(self.side, self.side)
        if transpose:
            return np.einsum("bij,bj->bi", blocks.transpose(0, 2, 1), matrix)
        return np.einsum("bij,bj->bi", blocks, matrix)

    def _permute(self, state: np.ndarray) -> np.ndarray:
        return state.reshape(self.side, self.side).T.reshape(-1)

    def _inverse_permute(self, state: np.ndarray) -> np.ndarray:
        return state.reshape(self.side, self.side).T.reshape(-1)

    def _apply_recurrent_unscaled(self, state: np.ndarray) -> np.ndarray:
        step1 = self._apply_blockdiag(self.blocks1, state, transpose=False).reshape(-1)
        step2 = self._permute(step1)
        step3 = self._apply_blockdiag(self.blocks2, step2, transpose=False).reshape(-1)
        return step3

    def _apply_recurrent_transpose_unscaled(self, state: np.ndarray) -> np.ndarray:
        step1 = self._apply_blockdiag(self.blocks2, state, transpose=True).reshape(-1)
        step2 = self._inverse_permute(step1)
        step3 = self._apply_blockdiag(self.blocks1, step2, transpose=True).reshape(-1)
        return step3


if __name__ == "__main__":
    evaluate_model(MonarchESN(), "Monarch ESN")
