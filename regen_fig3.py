"""Regenerate only Figure 3 (volatility-regime comparison) from the saved
walk-forward results CSV, without re-running the backtest or downloading data.

Mirrors the Figure 3 block in ``hdt/analysis.py`` so the standalone output and
the full-pipeline output stay identical.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from hdt.analysis import _save_figure, _setup_pubstyle

WF_CSV = "./outputs/walk_forward_results.csv"
OUTPUT_DIR = "./outputs"


def main() -> None:
    _setup_pubstyle()
    wf = pd.read_csv(WF_CSV, parse_dates=["test_start"])
    returns = wf["test_return"].values
    test_years = wf["test_start"].dt.year.values

    low = returns[test_years <= 2019]
    high = returns[test_years >= 2020]
    _, p_lh = stats.ttest_ind(low, high, equal_var=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    bp = ax.boxplot(
        [low * 100, high * 100],
        labels=["Low volatility\n(2015-2019)", "High volatility\n(2020-2024)"],
        patch_artist=True,
        showmeans=True,
        notch=True,
    )
    for patch, color in zip(bp["boxes"], ["lightskyblue", "lightcoral"]):
        patch.set_facecolor(color)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Quarterly return (%)")
    ax.set_title("(A) Return Distribution by Volatility Regime")
    ax.grid(alpha=0.3, axis="y")
    ax.annotate(
        f"two-sample $t$-test: $p = {p_lh:.2f}$\n(no significant separation)",
        xy=(0.5, 0.97), xycoords="axes fraction",
        ha="center", va="top", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85),
    )

    ax = axes[1]
    years = np.unique(test_years)
    year_means = np.array([returns[test_years == y].mean() * 100 for y in years])
    bar_colors = [
        "crimson" if y == 2022 else ("seagreen" if m > 0 else "indianred")
        for y, m in zip(years, year_means)
    ]
    ax.bar([str(y) for y in years], year_means, color=bar_colors,
           alpha=0.85, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean quarterly return (%)")
    ax.set_title("(B) Mean Return by Year (2022 bear market highlighted)")
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save_figure(fig, "figure3_regime_comparison", OUTPUT_DIR)
    print(f"low-vol mean={low.mean()*100:.2f}%  high-vol mean={high.mean()*100:.2f}%  p={p_lh:.2f}")


if __name__ == "__main__":
    main()
