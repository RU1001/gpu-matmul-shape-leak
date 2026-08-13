"""Timing sweep for the shape-leak experiment.

Measures execution time of Triton matmul and cuBLAS matmul across a range
of K values, holding M=N=1024 fixed. Writes raw timings to CSV.
"""

import argparse
import csv
from pathlib import Path

import torch

from src.matmul_kernel import matmul as triton_matmul


# --- CONFIG ---------------------------------------------------------------

# TODO: define the full shape list here as a list of K values.
# For dry-run mode, you'll want a shorter version — think about how to
# express that (a separate list? a slice of the main list?).
SHAPE_KS_FULL = [512, 1024, 1536, 2048, 3072, 4096]
SHAPE_KS_DRY = [1024, 2048]

M = 1024
N = 1024

WARMUP_TRIALS_FULL = 10
TIMED_TRIALS_FULL = 100

# TODO: pick reasonable values for dry-run (much smaller)
WARMUP_TRIALS_DRY = 3
TIMED_TRIALS_DRY = 5


# --- TIMING HELPER --------------------------------------------------------

def time_one_call(fn, a, b) -> float:
    """Time a single call to fn(a, b) using CUDA events. Returns ms."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# --- ONE SHAPE, ONE IMPL --------------------------------------------------

def measure_shape_impl(
    impl_name: str,
    fn,
    k: int,
    warmup_trials: int,
    timed_trials: int,
) -> list[dict]:
    """Return a list of dict rows, one per timed trial, ready for CSV."""
    rows = []

    # Warmup: fresh inputs each time, discard timings.
    # Reason: first launches include JIT compilation and cache cold-start.
    for _ in range(warmup_trials):
        a = torch.randn(M, k, device='cuda', dtype=torch.float16)
        b = torch.randn(k, N, device='cuda', dtype=torch.float16)
        fn(a, b)
    torch.cuda.synchronize()

    # we want each measurement to be independent).
    for trial in range(timed_trials):
        a = torch.randn(M, k, device='cuda', dtype=torch.float16)
        b = torch.randn(k, N, device='cuda', dtype=torch.float16)
        ms = time_one_call(fn, a, b)
        rows.append({
            'shape_k': k,
            'impl': impl_name,
            'trial': trial,
            'time_ms': ms,
        })

    return rows


# --- MAIN -----------------------------------------------------------------

def main():
    # TODO: set up argparse with a --dry-run flag.
    # TODO: pick the shape list and trial counts based on the flag.

    # For now, demo the pattern for one shape (K=1024) and both impls:
    k = 1024
    all_rows = []

    triton_rows = measure_shape_impl(
        impl_name='triton',
        fn=triton_matmul,
        k=k,
        warmup_trials=WARMUP_TRIALS_FULL,
        timed_trials=TIMED_TRIALS_FULL,
    )
    all_rows.extend(triton_rows)

    cublas_rows = measure_shape_impl(
        impl_name='cublas',
        fn=torch.matmul,
        k=k,
        warmup_trials=WARMUP_TRIALS_FULL,
        timed_trials=TIMED_TRIALS_FULL,
    )
    all_rows.extend(cublas_rows)

    # TODO: wrap the two calls above in a loop over your shape list.

    # Write CSV.
    out_dir = Path('experiments/shape_timing/data')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'timings.csv'

    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['shape_k', 'impl', 'trial', 'time_ms'])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_path}")


if __name__ == '__main__':
    main()
