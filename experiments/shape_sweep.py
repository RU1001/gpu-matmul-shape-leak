"""Timing sweep for the shape-leak experiment.

Sweeps K from 512 to 4096 (step=8) with M=N=1024 fixed.
Writes raw per-trial timings to experiments/data/timings.csv.
"""

import argparse
import csv
import random
from pathlib import Path

import torch

from src.matmul_kernel import matmul_kernel

import triton


# --- CONFIG ---------------------------------------------------------------

# grid: 512 to 4096 inclusive, step 8.  449 K values.
SHAPE_KS_FULL = list(range(512, 4097, 8))
SHAPE_KS_DRY = [512, 520, 528, 544, 1024, 1056]

M, N = 1024, 1024

WARMUP_TRIALS = 10
NUM_TRIALS = 100

WARMUP_TRIALS_DRY = 3
NUM_TRIALS_DRY = 5

# Triton tile config — from matmul_kernel.py
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32
GROUP_SIZE_M = 8
NUM_WARPS = 4
NUM_STAGES = 2


# --- TIMING HELPER --------------------------------------------------------

def time_one_call(fn) -> float:
    """Time a single zero-arg call using CUDA events.  Returns ms."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# --- TRITON RAW LAUNCH ----------------------------------------------------

def make_triton_launcher(a, b, c):
    """Return a zero-arg callable that launches the Triton kernel."""
    M_val, K_val = a.shape
    _, N_val = b.shape
    grid = (
        triton.cdiv(M_val, BLOCK_SIZE_M) * triton.cdiv(N_val, BLOCK_SIZE_N),
    )

    def launch():
        matmul_kernel[grid](
            a, b, c,
            M_val, N_val, K_val,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            ACTIVATION="",
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    return launch


def make_cublas_launcher(a, b, c):
    """Return a zero-arg callable that runs cuBLAS matmul into c."""
    def launch():
        torch.mm(a, b, out=c)
    return launch


# --- ONE SHAPE --------------------------------------------------

def measure_shape_impl(
    impl_name: str,
    make_launcher,
    k: int,
    warmup_trials: int,
    timed_trials: int,
) -> list[dict]:
    """Return a list of dict rows, one per timed trial."""
    # Preallocate all tensors once for this K.
    a = torch.randn(M, k, device='cuda', dtype=torch.float16)
    b = torch.randn(k, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)

    launcher = make_launcher(a, b, c)

    # Warmup: run kernel, discard timings.
    for _ in range(warmup_trials):
        launcher()
    torch.cuda.synchronize()

    # Timed trials.
    rows = []
    for trial in range(timed_trials):
        ms = time_one_call(launcher)
        rows.append({
            'shape_k': k,
            'impl': impl_name,
            'trial': trial,
            'time_ms': ms,
        })

    return rows


# --- MAIN -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Reduced shapes and trials for quick testing.')
    args = parser.parse_args()

    if args.dry_run:
        shape_list = list(SHAPE_KS_DRY)
        warmup = WARMUP_TRIALS_DRY
        timed = NUM_TRIALS_DRY
    else:
        shape_list = list(SHAPE_KS_FULL)
        warmup = WARMUP_TRIALS
        timed = NUM_TRIALS

    # Shuffle K order 
    random.seed(42)
    random.shuffle(shape_list)

    total = len(shape_list) * 2  # two impls per K
    all_rows = []

    for i, k in enumerate(shape_list):
        print(f"[{i+1}/{len(shape_list)}] K={k}")

        triton_rows = measure_shape_impl(
            impl_name='triton',
            make_launcher=make_triton_launcher,
            k=k,
            warmup_trials=warmup,
            timed_trials=timed,
        )
        all_rows.extend(triton_rows)

        cublas_rows = measure_shape_impl(
            impl_name='cublas',
            make_launcher=make_cublas_launcher,
            k=k,
            warmup_trials=warmup,
            timed_trials=timed,
        )
        all_rows.extend(cublas_rows)

    # Sort by (impl, shape_k, trial) before writing 
    all_rows.sort(key=lambda r: (r['impl'], r['shape_k'], r['trial']))

    # Write CSV.
    out_dir = Path('experiments/data')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'timings.csv'

    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['shape_k', 'impl', 'trial', 'time_ms'])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == '__main__':
    main()
