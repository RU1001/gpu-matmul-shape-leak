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


def tooth_amplitude(res, k, lo, hi):
    """Mean peak-to-trough within period-32 windows over [lo, hi).

    Step of 8 puts four samples per reduction tile, so each 4-sample
    window is one tooth. Averaged over a K-range this is a stable
    estimate of tooth height.
    """
    m = (k >= lo) & (k < hi)
    r = res[m]
    return np.mean([r[i:i + 4].ptp() for i in range(0, len(r) - 4, 4)])


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
    # Bottom row: residual vs K, linear and quadratic detrend overlaid.
    #   Linear detrend is what the post used; the quadratic overlay checks
    #   whether growing tooth height is real signal or leftover curvature
    #   the straight-line fit doesn't capture.
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

        # Linear and quadratic detrend.
        res_lin = mean - np.polyval(np.polyfit(k, mean, 1), k)
        res_quad = mean - np.polyval(np.polyfit(k, mean, 2), k)

        # Bottom row: both residuals overlaid.
        ax_bot = axes[1, col]
        ax_bot.plot(k, res_lin, linewidth=0.8, color='tab:orange',
                    label='linear detrend')
        ax_bot.plot(k, res_quad, linewidth=0.8, color='tab:blue', alpha=0.7,
                    label='quadratic detrend')
        ax_bot.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax_bot.set_xlabel('K (reduction dimension)')
        ax_bot.set_ylabel('Residual (ms)')
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

        # Quantify tooth height early vs late, both detrends, and print.
        # ratio ~1 under quadratic while linear ratio is high => the growth
        # was curvature, not signal. ratio high under both => real growth.
        k_max = k.max()
        early_lo, early_hi = k.min(), k.min() + 900
        late_lo, late_hi = k_max - 900, k_max
        print(f"\n[{impl}] tooth amplitude (peak-to-trough, ms):")
        for name, res in [('linear', res_lin), ('quadratic', res_quad)]:
            early = tooth_amplitude(res, k, early_lo, early_hi)
            late = tooth_amplitude(res, k, late_lo, late_hi)
            ratio = late / early if early else float('nan')
            print(f"  {name:9s} early={early:.4f}  late={late:.4f}  "
                  f"ratio={ratio:.2f}")

    fig.suptitle('Shape-dependent timing leak in Triton vs cuBLAS on T4')
    fig.tight_layout()

    out_path = PLOTS_DIR / 'timing_vs_shape.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Wrote plot to {out_path}")


if __name__ == '__main__':
    main()
