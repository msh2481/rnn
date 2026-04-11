import os
import random
from itertools import count
from multiprocessing import get_context

import pandas as pd
from solutions.rnn.lora import LoraESN
from solutions.rnn.monarch import MonarchESN
from utils import ScorerStepByStep

TRAIN_FILE = "datasets/train.parquet"
TEST_FILE = "datasets/test.parquet"
SCORER = None


def get_model():
    return MonarchESN(d=64)


def init_worker():
    global SCORER
    random.seed()
    SCORER = ScorerStepByStep(TEST_FILE)


def iteration(_):
    model = get_model()
    if hasattr(model, "train"):
        model.train(TRAIN_FILE, show_progress=False)
    results = SCORER.score(model, show_progress=False)
    return repr(model), results["mean_r2"]


def main():
    results = []
    iteration_num = 0
    output_csv = "opt_results.csv"
    process_count = 5
    print(f"Using {process_count} processes")

    ctx = get_context("spawn")
    with ctx.Pool(processes=process_count, initializer=init_worker) as pool:
        for name, r2 in pool.imap_unordered(iteration, count(), chunksize=2):
            iteration_num += 1
            print(f"Iteration {iteration_num:4}: {r2:.4f} {name:30}")
            results.append({"name": name, "comment": iteration_num, "r2": r2})
            if iteration_num % 10 == 0:
                df = pd.DataFrame(results)
                df = df.sort_values(by="r2", ascending=False).reset_index(drop=True)
                df.to_csv(output_csv, index=False)
                print(f"Saved {len(df)} results to {output_csv}")


if __name__ == "__main__":
    main()
