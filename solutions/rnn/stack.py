import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")
from solutions.rnn.es2n import ES2N
from solutions.simple.ema import EMA
from solutions.simple.poly import Poly
from utils import Stack


class StackESN(Stack):
    def __init__(
        self,
        memory=EMA(span=18),
        esn=ES2N(
            reservoir_size=64,
            beta=0.3559877553698028,
            nonlinearity="tanh",
            spectral_radius=0.9556453544583626,
            leak_rate=0.1324763897266856,
            input_scale=0.4,
            bias_scale=0.05,
            density=0.14398955615843656,
            ridge_alpha=0.07195134829894753,
        ),
    ):
        super().__init__(memory, esn)


if __name__ == "__main__":
    from utils import ScorerStepByStep

    train_file = f"{CURRENT_DIR}/../../datasets/train.parquet"
    test_file = f"{CURRENT_DIR}/../../datasets/test.parquet"
    model = StackESN()
    model.train(train_file)
    scorer = ScorerStepByStep(test_file)
    print("Testing StackESN...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")
    results = scorer.score(model)
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
