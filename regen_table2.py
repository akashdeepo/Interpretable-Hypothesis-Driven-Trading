"""Regenerate Table 2 (regime / sub-period performance) from the saved
walk-forward results CSV, without re-running the backtest.

Mirrors the Table 2 logic in ``hdt/analysis.py`` (sample-std / ddof=1 Sharpe,
manuscript sub-period structure) and prints the formatted rows used in the
manuscript so the LaTeX table can be kept in sync with the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hdt.analysis import _save_table

WF_CSV = "./outputs/walk_forward_results.csv"
OUTPUT_DIR = "./outputs"


def regime_block(returns: np.ndarray, mask: np.ndarray) -> dict:
    r = returns[mask]
    if r.size == 0:
        return {"N": 0, "Mean": np.nan, "Std": np.nan, "Win Rate": np.nan,
                "Best": np.nan, "Worst": np.nan, "Sharpe": np.nan}
    sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
    return {
        "N": int(r.size),
        "Mean": float(r.mean()),
        "Std": sd,
        "Win Rate": float((r > 0).sum() / r.size),
        "Best": float(r.max()),
        "Worst": float(r.min()),
        "Sharpe": float(r.mean() / sd * np.sqrt(4)) if sd > 0 else 0.0,
    }


def main() -> None:
    wf = pd.read_csv(WF_CSV, parse_dates=["test_start"])
    returns = wf["test_return"].values
    yr = wf["test_start"].dt.year.values

    rows = [
        ("Low volatility (2015-2019)", yr <= 2019),
        ("High volatility (2020-2024)", yr >= 2020),
        ("Pre-COVID (2017-2019)", (yr >= 2017) & (yr <= 2019)),
        ("COVID year (2020)", yr == 2020),
        ("Recovery / inflation (2021)", yr == 2021),
        ("Bear market (2022)", yr == 2022),
        ("Stabilisation (2023-2024)", yr >= 2023),
    ]

    table2 = pd.DataFrame(
        {
            "Regime": [name for name, _ in rows],
            **{
                k: [regime_block(returns, mask)[k] for _, mask in rows]
                for k in ["N", "Mean", "Std", "Win Rate", "Best", "Worst", "Sharpe"]
            },
        }
    )
    _save_table(table2, "table2_regime_performance", OUTPUT_DIR)

    print("\nManuscript-formatted rows (Mean %, Win %, SR(0)):")
    for name, mask in rows:
        b = regime_block(returns, mask)
        print(f"  {name:30s} & {b['N']:>2} & ${b['Mean']*100:+.2f}\\%$ "
              f"& {b['Win Rate']*100:.1f}\\% & ${b['Sharpe']:+.2f}$ \\\\")


if __name__ == "__main__":
    main()
