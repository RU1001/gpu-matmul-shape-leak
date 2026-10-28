"""Analyze timing sweep results and produce plots.

Reads experiments/data/timings.csv and writes:
- experiments/plots/timing_vs_shape.png
- experiments/data/summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("experiments/data/timings.csv")
PLOTS_DIR = Path("experiments/plots")
SUMMARY_PATH = Path("experiments/data/summary.csv")


def tooth_amplitude(res, k, lo, hi):
    """Estimate average tooth height over a K range."""
    mask = (k >= lo) & (k < hi)
    r = res[mask]

    return np.mean([
        np.ptp(r[i:i + 4])
        for i in range(0, len(r) - 4, 4)
    ])


def main():
    df = pd.read_csv(DATA_PATH)

    # Average repeated measurements for each implementation and K.
    summary = (
        df.groupby(["impl", "shape_k"])["time_ms"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["impl", "shape_k"])
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Wrote summary to {SUMMARY_PATH}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for col, impl in enumerate(["triton", "cublas"]):
        sub = summary[summary["impl"] == impl]

        k = sub["shape_k"].values
        mean = sub["mean"].values
        std = sub["std"].values

        # Raw timing.
        ax_top = axes[0, col]
        ax_top.plot(k, mean, linewidth=1.0)
        ax_top.fill_between(k, mean - std, mean + std, alpha=0.25)
        ax_top.set_ylabel("Execution time (ms)")
        ax_top.set_title(f"{impl} matmul, M=N=1024")
        ax_top.grid(True, alpha=0.3)

        # Remove the overall timing trend.
        linear_trend = np.polyval(np.polyfit(k, mean, 1), k)
        quadratic_trend = np.polyval(np.polyfit(k, mean, 2), k)

        res_linear = mean - linear_trend
        res_quadratic = mean - quadratic_trend

        # Residuals after detrending.
        ax_bot = axes[1, col]
        ax_bot.plot(
            k,
            res_linear,
            linewidth=0.8,
            label="linear detrend",
        )
        ax_bot.plot(
            k,
            res_quadratic,
            linewidth=0.8,
            alpha=0.7,
            label="quadratic detrend",
        )
        ax_bot.axhline(0, linewidth=0.5, linestyle="--")
        ax_bot.set_xlabel("K (reduction dimension)")
        ax_bot.set_ylabel("Residual (ms)")
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

        # Compare tooth height at the beginning and end of the sweep.
        k_max = k.max()
        early_lo = k.min()
        early_hi = k.min() + 900
        late_lo = k_max - 900
        late_hi = k_max

        print(f"\n[{impl}] tooth amplitude (peak-to-trough, ms):")

        for name, residual in [
            ("linear", res_linear),
            ("quadratic", res_quadratic),
        ]:
            early = tooth_amplitude(
                residual, k, early_lo, early_hi
            )
            late = tooth_amplitude(
                residual, k, late_lo, late_hi
            )
            ratio = late / early if early else float("nan")

            print(
                f"  {name:9s} "
                f"early={early:.4f} "
                f"late={late:.4f} "
                f"ratio={ratio:.2f}"
            )

    fig.suptitle("Shape-dependent timing leak in Triton vs cuBLAS on T4")
    fig.tight_layout()

    out_path = PLOTS_DIR / "timing_vs_shape.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
