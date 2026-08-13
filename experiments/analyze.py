"""Analyze timing sweep results and produce plots.

Reads experiments/data/timings.csv and writes:
- experiments/plots/timing_vs_shape.png (two-panel plot, Triton | cuBLAS)
- experiments/data/summary.csv (per-shape mean and std)
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path('experiments/data/timings.csv')
PLOTS_DIR = Path('experiments/plots')
SUMMARY_PATH = Path('experiments/data/summary.csv')


def main():
    df = pd.read_csv(DATA_PATH)

    # Per-(shape, impl) summary: mean, std, count.
    summary = (
        df.groupby(['impl', 'shape_k'])['time_ms']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .sort_values(['impl', 'shape_k'])
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Wrote summary to {SUMMARY_PATH}")
    print(summary.to_string(index=False))

    # Two-panel plot: separate y-axes because scales differ by ~275x.
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_triton, ax_cublas) = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, impl in [(ax_triton, 'triton'), (ax_cublas, 'cublas')]:
        sub = summary[summary['impl'] == impl]
        ax.errorbar(
            sub['shape_k'], sub['mean'], yerr=sub['std'],
            marker='o', capsize=4, linewidth=1.5,
        )
        ax.set_xlabel('K (reduction dimension)')
        ax.set_ylabel('Execution time (ms)')
        ax.set_title(f'{impl} matmul, M=N=1024')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Shape-dependent timing leak in Triton vs cuBLAS on T4', y=1.02)
    fig.tight_layout()

    out_path = PLOTS_DIR / 'timing_vs_shape.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Wrote plot to {out_path}")


if __name__ == '__main__':
    main()
