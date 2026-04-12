import time

import torch
from torch.utils.data import DataLoader

from src.dl.dl_base import SequenceDataset


def bench_loader(dataset, batch_size=32, num_workers=0, pin_memory=False, n_epochs=3):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # warmup
    for batch in loader:
        pass

    t0 = time.perf_counter()
    for _ in range(n_epochs):
        for batch in loader:
            pass
    elapsed = time.perf_counter() - t0

    batches = len(loader) * n_epochs
    print(
        f"  workers={num_workers:2d}  pin_memory={pin_memory!s:5s}  "
        f"batch_size={batch_size:4d}  "
        f"{elapsed:.3f}s total  {elapsed/batches*1000:.2f}ms/batch  "
        f"{elapsed/n_epochs*1000:.1f}ms/epoch"
    )
    return elapsed


if __name__ == "__main__":
    print("Loading dataset...")
    ds = SequenceDataset()
    print(f"Dataset: {len(ds)} sequences, shape {ds.data.shape}")
    print()

    batch_size = 32

    print(f"--- batch_size={batch_size} ---")
    for num_workers in [0, 1, 2, 4]:
        for pin_memory in [False, True]:
            bench_loader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    print()
    print(f"--- batch_size=64 ---")
    for num_workers in [0, 1, 2, 4]:
        for pin_memory in [False, True]:
            bench_loader(ds, batch_size=64, num_workers=num_workers, pin_memory=pin_memory)
