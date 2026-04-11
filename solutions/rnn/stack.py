import sys

from pandas import os
from solutions.rnn.es2n import ES2N
from solutions.simple.ema import EMA
from solutions.simple.poly import Poly
from utils import Stack

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")


class StackESN(Stack):
    def __init__(
        self,
        memory: EMA | Poly,
        esn: ES2N,
    ):
        super().__init__(memory, esn)


if __name__ == "__main__":
    from utils import ScorerStepByStep

    test_file = f"{CURRENT_DIR}/../../datasets/test.parquet"
    model = StackESN(EMA(), ES2N())
    scorer = ScorerStepByStep(test_file)
    print("Testing StackESN...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")
    results = scorer.score(model)
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
