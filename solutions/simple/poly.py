import os
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")

from utils import DataPoint


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
    from utils import ScorerStepByStep

    test_file = f"{CURRENT_DIR}/../../datasets/test.parquet"
    model = Poly(d=6, p=4 / 3)
    print(model)
    scorer = ScorerStepByStep(test_file)
    print("Testing simple model with polynomial moving average...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")
    results = scorer.score(model)
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")
    print(f"MSE score: {results['mse_score']:.6f}")
