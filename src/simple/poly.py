import numpy as np

from src.utils import DataPoint


class Poly:
    def __init__(self, d=None, p=None):
        self.d = d
        if self.d is None:
            self.d = np.random.lognormal(mean=np.log(1), sigma=np.log(10))

        self.p = p
        if self.p is None:
            self.p = 1 + np.random.exponential(scale=1.0)

        self.current_seq_ix = None
        self.sequence_history = []

    def __repr__(self):
        return f"Poly(d={self.d}, p={self.p})"

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        self.sequence_history.append(data_point.state.copy())

        history = np.array(self.sequence_history[::-1][:100])
        indices = np.arange(len(history))
        weights = (indices + self.d) ** (-self.p)
        weights = weights / weights.sum()
        return np.sum(history * weights[:, None], axis=0)


if __name__ == "__main__":
    from src.utils import train_and_eval
    train_and_eval(Poly(d=6, p=4 / 3))
