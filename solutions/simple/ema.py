import os
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")

from utils import DataPoint


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
    from utils import ScorerStepByStep

    test_file = f"{CURRENT_DIR}/../../datasets/test.parquet"
    model = EMA(span=18)
    print(model)
    scorer = ScorerStepByStep(test_file)
    print("Testing simple model with moving average...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")
    results = scorer.score(model)
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")
    print(f"MSE score: {results['mse_score']:.6f}")
