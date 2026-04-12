import numpy as np

from src.utils import DataPoint


class Mu100_3:
    def __init__(self):
        self.current_seq_ix = None
        self.sequence_history = []

    def __repr__(self):
        return "Mu100_3()"

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
        self.sequence_history.append(data_point.state.copy())
        mu100 = np.mean(self.sequence_history[-100:], axis=0)
        mu3 = np.mean(self.sequence_history[-3:], axis=0)
        return mu100 + 0.3 * (mu3 - mu100)


if __name__ == "__main__":
    from src.utils import train_and_eval
    train_and_eval(Mu100_3())
