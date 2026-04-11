import os
import sys
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")

from utils import DataPoint


class PredictionModel:
    def __init__(self):
        self.current_seq_ix = None
        self.sequence_history = []

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
        self.sequence_history.append(data_point.state.copy())
        if not data_point.need_prediction:
            return None
        mu100 = np.mean(self.sequence_history[-100:], axis=0)
        mu3 = np.mean(self.sequence_history[-3:], axis=0)
        return mu100 + 0.3 * (mu3 - mu100) 


if __name__ == "__main__":
    from utils import ScorerStepByStep
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"
    model = PredictionModel()
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