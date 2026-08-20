"""Analyze timing sweep results and produce plots.

Reads experiments/data/timings.csv and writes:
- experiments/plots/timing_vs_shape.png
- experiments/data/summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path('experiments/data/timings.csv')
PLOTS_DIR = Path('experiments/plots')
SUMMARY_PATH = Path('experiments/data/summary.csv')


def main():
    df = pd.read_csv(DATA_PATH)

    # Per-(shape, impl) summary.
    summary = (
        df.groupby(['impl', 'shape_k'])['time_ms']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .sort_values(['impl', 'shape_k'])
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Wrote summary to {SUMMARY_PATH}")

    # --- Plot: 2 rows x 2 cols ---
    # Top row: raw timing vs K.
    # Bottom row: residual (timing minus linear fit) to isolate staircase.
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [2, 1]})

    for col, impl in enumerate(['triton', 'cublas']):
        sub = summary[summary['impl'] == impl]
        k = sub['shape_k'].values
        mean = sub['mean'].values
        std = sub['std'].values

        # Top row: raw timing with shaded error band.
        ax_top = axes[0, col]
        ax_top.plot(k, mean, linewidth=1.0)
        ax_top.fill_between(k, mean - std, mean + std, alpha=0.25)
        ax_top.set_ylabel('Execution time (ms)')
        ax_top.set_title(f'{impl} matmul, M=N=1024')
        ax_top.grid(True, alpha=0.3)

        # Linear fit for residual.
        coeffs = np.polyfit(k, mean, 1)
        trend = np.polyval(coeffs, k)

        # Bottom row: residual (mean - linear fit).
        ax_bot = axes[1, col]
        ax_bot.plot(k, mean - trend, linewidth=0.8, color='tab:orange')
        ax_bot.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax_bot.set_xlabel('K (reduction dimension)')
        ax_bot.set_ylabel('Residual (ms)')
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle('Shape-dependent timing leak in Triton vs cuBLAS on T4')
    fig.tight_layout()

    out_path = PLOTS_DIR / 'timing_vs_shape.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Wrote plot to {out_path}")


if __name__ == '__main__':
    main()
