import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")

from solutions.rnn.es2n import ES2N
from solutions.simple.ema import EMA
from solutions.simple.poly import Poly
from utils import Bag, Stack


class BagESN(Bag):
    def __init__(self, ridge_alpha=None):
        models = [
            ES2N(reservoir_size=64),
            ES2N(reservoir_size=64),
            Stack(EMA(span=18), ES2N(reservoir_size=64)),
            Stack(Poly(d=6.0, p=4 / 3), ES2N(reservoir_size=64)),
            Stack(EMA(span=100), ES2N(reservoir_size=64)),
            Stack(Poly(d=20.0, p=1.0), ES2N(reservoir_size=64)),
        ]
        super().__init__(models, ridge_alpha=ridge_alpha)


if __name__ == "__main__":
    from utils import ScorerStepByStep

    train_file = f"{CURRENT_DIR}/../../datasets/train.parquet"
    test_file = f"{CURRENT_DIR}/../../datasets/test.parquet"
    model = BagESN()
    print(model)
    print("Training BagESN...")
    model.train(train_file)
    scorer = ScorerStepByStep(test_file)
    print("Testing BagESN...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")
    results = scorer.score(model)
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")
    print(f"MSE score: {results['mse_score']:.6f}")
