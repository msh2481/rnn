import numpy as np

from src.utils import DataPoint


class EMA:
    def __init__(self, span=None):
        self.span = span
        if self.span is None:
            self.span = np.random.lognormal(mean=np.log(30), sigma=np.log(10))

        self.alpha = 1 / (self.span + 1)
        self.current_seq_ix = None
        self.ema = None

    def __repr__(self):
        return f"EMA(span={self.span})"

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.ema = None
        if self.ema is None:
            self.ema = np.zeros_like(data_point.state)
        else:
            self.ema = self.alpha * data_point.state + (1 - self.alpha) * self.ema
        return self.ema.copy()


if __name__ == "__main__":
    from src.utils import train_and_eval
    train_and_eval(EMA(span=18))
