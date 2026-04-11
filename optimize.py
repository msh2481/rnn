import os
import random
import time
from itertools import count
from multiprocessing import get_context

import pandas as pd
from solutions.rnn.es2n import ES2N
from solutions.rnn.esn import ESN
from utils import ScorerStepByStep

TRAIN_FILE = "datasets/train.parquet"
TEST_FILE = "datasets/test.parquet"
SCORER = None


def get_model():
    return ES2N(reservoir_size=64)


def init_worker():
    global SCORER
    random.seed()
    SCORER = ScorerStepByStep(TEST_FILE)


def iteration(_):
    start_time = time.perf_counter()
    model = get_model()
    if hasattr(model, "train"):
        model.train(TRAIN_FILE, show_progress=False)
    results = SCORER.score(model, show_progress=False)
    elapsed_time = time.perf_counter() - start_time
    return repr(model), results["mean_r2"], elapsed_time


def main():
    results = []
    iteration_num = 0
    output_csv = "opt_results.csv"
    process_count = 5
    batch_size = 10
    last_batch_time = time.perf_counter()
    print(f"Using {process_count} processes")

    ctx = get_context("spawn")
    with ctx.Pool(processes=process_count, initializer=init_worker) as pool:
        for name, r2, elapsed_time in pool.imap_unordered(
            iteration, count(), chunksize=2
        ):
            iteration_num += 1
            print(
                f"Iteration {iteration_num:4}: {r2:.4f} "
                f"({elapsed_time:.3f}s) {name:30}"
            )
            results.append(
                {
                    "name": name,
                    "comment": iteration_num,
                    "r2": r2,
                    "time": elapsed_time,
                }
            )
            if iteration_num % batch_size == 0:
                now = time.perf_counter()
                avg_seconds_per_result = (now - last_batch_time) / batch_size
                last_batch_time = now
                df = pd.DataFrame(results)
                df = df.sort_values(by="r2", ascending=False).reset_index(drop=True)
                df.to_csv(output_csv, index=False)
                print(
                    f"Saved {len(df)} results to {output_csv} "
                    f"({avg_seconds_per_result:.3f}s/result over last batch)"
                )


if __name__ == "__main__":
    main()
