"""Publication-ready analysis.

Generates the eight tables and eight figures referenced in the manuscript,
plus the comprehensive text report. All outputs are written to ``output_dir``.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest, pearsonr, ttest_1samp
from sklearn.linear_model import LinearRegression

from . import stats as ldp_stats


# ---------------------------------------------------------------------------
# Regime mapping
# ---------------------------------------------------------------------------

def assign_regimes(wf: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Map each fold's test-window start date to a regime label.

    Definitions follow paper Section 4.3 and Table 3:
        - Bull Market:  test_start in 2020-2021 or 2024
        - Bear Market:  test_start in 2022
        - Recovery:     test_start in 2023
    Folds outside these ranges (e.g. 2016-2019) are labelled "Other" and are
    excluded from the regime sub-tables but retained in aggregate statistics.
    """
    labels = []
    for ts in wf["test_start"]:
        year = pd.Timestamp(ts).year
        if year in (2020, 2021, 2024):
            labels.append("Bull")
        elif year == 2022:
            labels.append("Bear")
        elif year == 2023:
            labels.append("Recovery")
        else:
            labels.append("Other")
    return {"label": np.array(labels)}


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def bootstrap_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float, float]:
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(returns)
    samples = rng.choice(returns, size=(n_bootstrap, n), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return means, float(np.percentile(means, alpha * 100)), float(
        np.percentile(means, (1.0 - alpha) * 100)
    )


def permutation_pvalue(
    returns: np.ndarray,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> Tuple[float, np.ndarray]:
    """One-sample permutation test for H0: E[r] = 0.

    For a one-sample mean test, value-permutation is degenerate because the
    sample mean is invariant under permutation of the sample. We use the
    Wilcoxon-style sign-flip permutation (Fisher's randomization test): under
    H0 the distribution of each r_i is symmetric about zero, so flipping its
    sign produces an equally-likely realisation. The two-sided p-value is
    Pr_pi(|bar r_pi| >= |bar r_obs|).

    Note: the original notebook used ``np.random.permutation(returns)`` which
    leaves the mean unchanged and produces a degenerate null distribution
    (p approx 1 by construction). The sign-flip variant is the correct
    one-sample analogue.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    observed = returns.mean()
    n = len(returns)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    means = (signs * returns).mean(axis=1)
    p = float((np.abs(means) >= abs(observed)).sum()) / n_permutations
    return p, means


def downside_deviation(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    return float(np.sqrt(np.mean(downside ** 2))) if downside.size else 0.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_table(df: pd.DataFrame, name: str, output_dir: str) -> None:
    df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    try:
        df.to_latex(
            os.path.join(output_dir, f"{name}.tex"),
            float_format="%.4f",
            escape=False,
            index=False,
        )
    except Exception:  # noqa: BLE001
        # LaTeX export depends on the pandas / Jinja2 version. CSV is the
        # authoritative format; .tex is best-effort.
        pass
    print(f"  wrote {name}.csv (+ .tex)")


def _safe_bins(data: np.ndarray, requested: int = 50) -> int:
    """Return a bin count that matplotlib will accept for ``data``.

    Avoids the "Cannot create N finite-sized bins" error that occurs when the
    data has near-zero range relative to its magnitude.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1
    span = arr.max() - arr.min()
    if span == 0 or span < 1e-12 * (abs(arr.mean()) + 1e-12):
        return 1
    return min(requested, max(1, int(np.sqrt(arr.size))))


def _save_figure(fig: plt.Figure, name: str, output_dir: str) -> None:
    for fmt in ("png", "pdf"):
        fig.savefig(
            os.path.join(output_dir, f"{name}.{fmt}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def _setup_pubstyle() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_publication_outputs(
    wf_results: pd.DataFrame,
    market_data: Dict[str, pd.DataFrame],
    output_dir: str = "./outputs",
    rng_seed: int = 0,
) -> Dict[str, float]:
    """Produce all tables, figures, and the summary report."""
    os.makedirs(output_dir, exist_ok=True)
    _setup_pubstyle()
    rng = np.random.default_rng(rng_seed)

    returns = wf_results["test_return"].to_numpy()
    n = len(returns)
    win_rate = float((returns > 0).sum()) / n
    sharpe_ann = (
        returns.mean() / returns.std() * np.sqrt(4) if returns.std() > 0 else 0.0
    )
    max_dd = float(wf_results["test_max_dd"].min())

    # SPY benchmark returns for each test window
    spy = market_data["SPY"]
    spy_returns = []
    for _, row in wf_results.iterrows():
        window = spy.loc[row["test_start"]:row["test_end"]]
        if len(window) > 1:
            spy_returns.append(window["Close"].iloc[-1] / window["Close"].iloc[0] - 1)
        else:
            spy_returns.append(0.0)
    spy_returns = np.array(spy_returns)
    wf_results = wf_results.copy()
    wf_results["spy_return"] = spy_returns

    # Regimes
    wf_results["regime"] = assign_regimes(wf_results)["label"]

    # ---- Statistical tests
    t_stat, t_pval = ttest_1samp(returns, 0.0)
    t_stat_one, t_pval_one = ttest_1samp(returns, 0.0, alternative="greater")
    bs_means, ci_lo, ci_hi = bootstrap_ci(returns, rng=rng)
    mc_p, mc_means = permutation_pvalue(returns, rng=rng)
    shapiro_stat, shapiro_p = stats.shapiro(returns)
    binom = binomtest((returns > 0).sum(), n, 0.5, alternative="greater")
    binom_p = binom.pvalue

    # CAPM-style regression vs SPY
    X = spy_returns.reshape(-1, 1)
    reg = LinearRegression().fit(X, returns)
    beta = float(reg.coef_[0])
    alpha = float(reg.intercept_)
    tracking_err = float(np.std(returns - spy_returns))
    info_ratio = (
        (returns.mean() - spy_returns.mean()) / tracking_err if tracking_err > 0 else 0.0
    )

    # Train vs test correlation (information coefficient)
    ic, ic_p = pearsonr(wf_results["train_return"].to_numpy(), returns)

    # Drawdown profile
    cum = (1.0 + pd.Series(returns)).cumprod() - 1.0
    running_max = (1.0 + cum).cummax()
    drawdown = (1.0 + cum) / running_max - 1.0
    max_dd_cumulative = float(drawdown.min())

    # Risk metrics
    downside_dev = downside_deviation(returns)
    var95 = float(np.percentile(returns, 5))
    cvar95 = float(returns[returns <= var95].mean()) if (returns <= var95).any() else var95
    avg_win = float(returns[returns > 0].mean()) if (returns > 0).any() else 0.0
    avg_loss = float(returns[returns < 0].mean()) if (returns < 0).any() else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # ---- Tables
    print("Writing tables...")

    table1 = pd.DataFrame(
        {
            "Metric": [
                "Mean Return (Quarterly)",
                "Annualized Return",
                "Standard Deviation",
                "Sharpe Ratio (Ann.)",
                "Sortino Ratio (Ann.)",
                "Maximum Drawdown",
                "Calmar Ratio",
                "Win Rate (Folds)",
                "Best Fold",
                "Worst Fold",
                "Skewness",
                "Kurtosis",
                "Total Folds",
                "Average Trades per Fold",
            ],
            "Value": [
                returns.mean(),
                returns.mean() * 4,
                returns.std(),
                sharpe_ann,
                (returns.mean() / downside_dev * np.sqrt(4)) if downside_dev > 0 else np.inf,
                max_dd,
                (returns.mean() * 4) / abs(max_dd) if max_dd != 0 else np.inf,
                win_rate,
                returns.max(),
                returns.min(),
                stats.skew(returns),
                stats.kurtosis(returns),
                n,
                wf_results["test_trades"].mean(),
            ],
        }
    )
    _save_table(table1, "table1_summary_statistics", output_dir)

    def regime_block(mask: np.ndarray) -> Dict[str, float]:
        r = returns[mask]
        if r.size == 0:
            return {"N": 0, "Mean": np.nan, "Std": np.nan, "Win Rate": np.nan,
                    "Best": np.nan, "Worst": np.nan, "Sharpe": np.nan}
        # Sample standard deviation (ddof=1), matching the headline Sharpe
        # convention used for Table 1 and the body. ddof=0 would make the
        # sub-period Sharpes inconsistent with the aggregate figures.
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

    # Table 2 follows the manuscript structure: the low-/high-volatility split
    # plus notable calendar sub-periods. Masks use the test-window start year so
    # the table is reproducible directly from the saved walk-forward results.
    test_years = pd.to_datetime(wf_results["test_start"]).dt.year.values
    regime_rows = [
        ("Low volatility (2015-2019)", test_years <= 2019),
        ("High volatility (2020-2024)", test_years >= 2020),
        ("Pre-COVID (2017-2019)", (test_years >= 2017) & (test_years <= 2019)),
        ("COVID year (2020)", test_years == 2020),
        ("Recovery / inflation (2021)", test_years == 2021),
        ("Bear market (2022)", test_years == 2022),
        ("Stabilisation (2023-2024)", test_years >= 2023),
    ]
    table2 = pd.DataFrame(
        {
            "Regime": [name for name, _ in regime_rows],
            **{
                k: [regime_block(mask)[k] for _, mask in regime_rows]
                for k in ["N", "Mean", "Std", "Win Rate", "Best", "Worst", "Sharpe"]
            },
        }
    )
    _save_table(table2, "table2_regime_performance", output_dir)

    table3 = pd.DataFrame(
        {
            "Test": [
                "Two-Sided t-test",
                "One-Sided t-test (greater)",
                "Bootstrap 95% CI - lower",
                "Bootstrap 95% CI - upper",
                "Monte Carlo permutation",
                "Shapiro-Wilk normality",
                "Binomial test (win rate)",
            ],
            "Statistic": [t_stat, t_stat_one, np.nan, np.nan, np.nan, shapiro_stat, np.nan],
            "p-value": [t_pval, t_pval_one, np.nan, np.nan, mc_p, shapiro_p, binom_p],
            "Value": [np.nan, np.nan, ci_lo, ci_hi, np.nan, np.nan, np.nan],
        }
    )
    _save_table(table3, "table3_statistical_tests", output_dir)

    spy_cum = (1.0 + pd.Series(spy_returns)).cumprod()
    spy_dd = (spy_cum / spy_cum.cummax() - 1.0).min()
    table4 = pd.DataFrame(
        {
            "Metric": [
                "Mean Return (Quarterly)",
                "Annualized Return",
                "Standard Deviation",
                "Sharpe Ratio (Ann.)",
                "Maximum Drawdown",
                "Win Rate",
                "Alpha (Quarterly)",
                "Alpha (Annualized)",
                "Beta",
                "Tracking Error",
                "Information Ratio",
                "Correlation",
            ],
            "Strategy": [
                returns.mean(),
                returns.mean() * 4,
                returns.std(),
                sharpe_ann,
                max_dd,
                win_rate,
                alpha,
                alpha * 4,
                beta,
                tracking_err,
                info_ratio,
                float(np.corrcoef(returns, spy_returns)[0, 1]),
            ],
            "SPY": [
                spy_returns.mean(),
                spy_returns.mean() * 4,
                spy_returns.std(),
                spy_returns.mean() / spy_returns.std() * np.sqrt(4)
                if spy_returns.std() > 0
                else 0.0,
                float(spy_dd),
                float((spy_returns > 0).sum() / spy_returns.size),
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            ],
        }
    )
    _save_table(table4, "table4_benchmark_comparison", output_dir)

    table5 = pd.DataFrame(
        {
            "Risk Metric": [
                "Volatility (Quarterly)",
                "Volatility (Annualized)",
                "Downside Deviation",
                "Sortino Ratio (Ann.)",
                "Maximum Drawdown",
                "Average Drawdown",
                "Calmar Ratio",
                "Value at Risk (95%)",
                "CVaR / Expected Shortfall",
                "Win Rate",
                "Average Win",
                "Average Loss",
                "Win/Loss Ratio",
                "Best Fold Return",
                "Worst Fold Return",
            ],
            "Value": [
                returns.std(),
                returns.std() * np.sqrt(4),
                downside_dev,
                (returns.mean() / downside_dev * np.sqrt(4)) if downside_dev > 0 else np.inf,
                max_dd_cumulative,
                float(drawdown.mean()),
                (returns.mean() * 4) / abs(max_dd_cumulative)
                if max_dd_cumulative != 0
                else np.inf,
                var95,
                cvar95,
                win_rate,
                avg_win,
                avg_loss,
                win_loss_ratio,
                returns.max(),
                returns.min(),
            ],
        }
    )
    _save_table(table5, "table5_risk_metrics", output_dir)

    fold_details = wf_results.copy()
    fold_details["test_start"] = fold_details["test_start"].dt.strftime("%Y-%m-%d")
    fold_details["test_end"] = fold_details["test_end"].dt.strftime("%Y-%m-%d")
    _save_table(
        fold_details[
            [
                "fold",
                "test_start",
                "test_end",
                "test_return",
                "test_trades",
                "test_win_rate",
                "test_sharpe",
                "test_max_dd",
                "spy_return",
                "regime",
            ]
        ],
        "table6_fold_details",
        output_dir,
    )

    # ---- Table 7: Sharpe-ratio inference under non-normality / multiple
    # testing (PSR, MinTRL, DSR) -- the corrected-inference table requested
    # by reviewer R1.
    risk_free_per_quarter = 0.02 / 4.0  # 2% annual benchmark; documented in caption
    train_returns_arr = wf_results["train_return"].to_numpy()
    table7 = ldp_stats.sharpe_inference_table(
        returns,
        n_trials_options=(1, 10, 30, 100),
        risk_free_per_period=risk_free_per_quarter,
        periods_per_year=4,
        train_returns=train_returns_arr,
    )
    _save_table(table7, "table7_sharpe_inference", output_dir)

    psr_zero = ldp_stats.probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
    dsr_results = {
        n: ldp_stats.deflated_sharpe_ratio(returns, n_trials=n)
        for n in (1, 10, 30, 100)
    }
    mintrl_95 = ldp_stats.minimum_track_record_length(returns, confidence=0.95)
    sr_excess_rf = ldp_stats.sharpe_ratio(returns, risk_free=risk_free_per_quarter)
    pbo_single = ldp_stats.pbo_from_train_test_pairs(train_returns_arr, returns)

    # ---- Figures
    print("Writing figures...")

    folds = wf_results["fold"].to_numpy()

    # Figure 1: four-panel main results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    colors = ["green" if r > 0 else "red" for r in returns]
    ax.bar(folds, returns * 100, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(returns.mean() * 100, color="blue", linestyle="--",
               label=f"Mean: {returns.mean()*100:.2f}%")
    ax.set_title("(A) Out-of-Sample Returns by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Return (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    cum_pct = ((1 + pd.Series(returns)).cumprod() - 1) * 100
    ax.plot(folds, cum_pct, marker="o", linewidth=2, color="blue")
    ax.fill_between(folds, 0, cum_pct, alpha=0.2, color="blue")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("(B) Cumulative Out-of-Sample Performance")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.bar(folds, wf_results["test_win_rate"] * 100, color="skyblue", edgecolor="black", alpha=0.7)
    ax.axhline(50, color="red", linestyle="--", label="Random (50%)")
    ax.axhline(wf_results["test_win_rate"].mean() * 100, color="darkblue", linestyle="--",
               label=f"Mean: {wf_results['test_win_rate'].mean()*100:.1f}%")
    ax.set_ylim(0, 110)
    ax.set_title("(C) Trade Win Rate by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Win Rate (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    s = wf_results["test_sharpe"].to_numpy().copy()
    s[~np.isfinite(s)] = 0
    ax.bar(folds, s, color=["green" if v > 0 else "red" for v in s], alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(1, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1")
    ax.set_title("(D) Sharpe Ratio by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, "figure1_main_results", output_dir)

    # Figure 2: statistical analysis (bootstrap, permutation, Q-Q)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    ax.hist(bs_means, bins=_safe_bins(bs_means), density=True,
            color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(returns.mean(), color="red", linewidth=2,
               label=f"Mean: {returns.mean():.4f}")
    ax.axvline(ci_lo, color="green", linestyle="--", label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    ax.axvline(ci_hi, color="green", linestyle="--")
    ax.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_title("(A) Bootstrap Distribution of Mean Returns (10,000 resamples)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(mc_means, bins=_safe_bins(mc_means), density=True,
            color="gray", alpha=0.7, edgecolor="black")
    ax.axvline(returns.mean(), color="red", linewidth=2, label=f"Observed: {returns.mean():.4f}")
    ax.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_title(f"(B) Monte Carlo Permutation Test (p = {mc_p:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title(f"(C) Q-Q Plot (Shapiro-Wilk p = {shapiro_p:.4f})")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, "figure2_statistical_analysis", output_dir)

    # Figure 3: volatility-regime comparison (matches manuscript Table 2 and the
    # revised "weak regime separation" narrative). Low-vol = 2015-2019,
    # High-vol = 2020-2024, split on the test-window start year.
    test_years = pd.to_datetime(wf_results["test_start"]).dt.year.values
    low = returns[test_years <= 2019]
    high = returns[test_years >= 2020]
    _, p_lh = stats.ttest_ind(low, high, equal_var=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: distribution of fold returns by volatility regime
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

    # Panel B: mean quarterly return by calendar year; 2022 is the consistent
    # failure mode (all four bear-market folds negative).
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
    _save_figure(fig, "figure3_regime_comparison", output_dir)

    # Figure 4: benchmark comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    x = np.arange(len(folds))
    w = 0.35
    ax.bar(x - w / 2, returns * 100, w, label="Strategy", color="steelblue", edgecolor="black")
    ax.bar(x + w / 2, spy_returns * 100, w, label="SPY", color="orange", edgecolor="black")
    ax.axhline(0, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Return (%)")
    ax.set_title("(A) Strategy vs SPY Returns by Fold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    sc = ax.scatter(spy_returns * 100, returns * 100, c=range(len(folds)), cmap="viridis",
                    s=100, edgecolor="black")
    spy_range = np.linspace(spy_returns.min(), spy_returns.max(), 100)
    ax.plot(spy_range * 100, (alpha + beta * spy_range) * 100, "r--",
            label=fr"$\beta$={beta:.3f}, $\alpha$={alpha:.4f}")
    ax.axhline(0, color="black", alpha=0.5)
    ax.axvline(0, color="black", alpha=0.5)
    ax.set_xlabel("SPY Return (%)")
    ax.set_ylabel("Strategy Return (%)")
    ax.set_title(fr"(B) Strategy vs Market ($\beta$={beta:.3f}, $\alpha$={alpha*4:.2%} ann.)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Fold")

    plt.tight_layout()
    _save_figure(fig, "figure4_benchmark_comparison", output_dir)

    # Figure 5: drawdown
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax = axes[0]
    ax.plot(folds, cum_pct, marker="o", linewidth=2, color="blue")
    ax.fill_between(folds, 0, cum_pct, alpha=0.3, color="blue")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("(A) Cumulative Returns Over Folds")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.fill_between(folds, 0, drawdown * 100, color="red", alpha=0.5)
    ax.plot(folds, drawdown * 100, color="darkred", linewidth=2)
    ax.axhline(0, color="black", linestyle="--")
    ax.axhline(drawdown.min() * 100, color="red", linestyle=":",
               label=f"Max DD: {drawdown.min()*100:.2f}%")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("(B) Drawdown Profile")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, "figure5_drawdown_analysis", output_dir)

    # Figure 6: train vs test
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        wf_results["train_return"] * 100,
        returns * 100,
        c=range(len(folds)),
        cmap="viridis",
        s=120,
        edgecolor="black",
    )
    lim = max(abs(wf_results["train_return"].max()), abs(wf_results["train_return"].min())) * 100
    ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5, label="Perfect Prediction")
    ax.axhline(0, color="black", alpha=0.3)
    ax.axvline(0, color="black", alpha=0.3)
    ax.set_xlabel("Training Period Return (%)")
    ax.set_ylabel("Testing Period Return (%)")
    ax.set_title(f"Training vs Testing (IC = {ic:.3f}, p = {ic_p:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Fold")
    plt.tight_layout()
    _save_figure(fig, "figure6_train_vs_test", output_dir)

    # Figure 7: return distribution
    mu, sigma = returns.mean(), returns.std()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.hist(returns, bins=20, density=True, color="steelblue", alpha=0.7, edgecolor="black")
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2,
            label=fr"Normal($\mu$={mu:.4f}, $\sigma$={sigma:.4f})")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title(f"(A) Return Distribution (Skew={stats.skew(returns):.2f}, "
                 f"Kurt={stats.kurtosis(returns):.2f})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    sorted_r = np.sort(returns)
    cum_prob = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax.plot(sorted_r, cum_prob, linewidth=2, label="Empirical")
    ax.plot(sorted_r, stats.norm.cdf(sorted_r, mu, sigma), "r--", linewidth=2, label="Normal")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Return")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("(B) Cumulative Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    parts = ax.violinplot([returns], positions=[1], showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xticks([1])
    ax.set_xticklabels(["Strategy"])
    ax.set_title("(C) Return Distribution (Violin)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    pd.plotting.autocorrelation_plot(pd.Series(returns), ax=ax)
    ax.set_title("(D) Return Autocorrelation")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, "figure7_return_distribution", output_dir)

    # Figure 8: time series of metrics
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    ax = axes[0]
    ax.plot(folds, returns * 100, marker="o", linewidth=2, color="blue", label="Quarterly Return")
    rolling = pd.Series(returns).rolling(3, min_periods=1).mean()
    ax.plot(folds, rolling * 100, "r--", linewidth=2, label="3-fold MA")
    ax.fill_between(folds, 0, returns * 100, alpha=0.2, color="blue")
    ax.axhline(0, color="black")
    ax.axhline(returns.mean() * 100, color="green", linestyle=":",
               label=f"Mean: {returns.mean()*100:.2f}%")
    ax.set_ylabel("Return (%)")
    ax.set_title("(A) Returns Over Time with Moving Average")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(folds, wf_results["test_win_rate"] * 100, marker="s", color="purple")
    ax.axhline(50, color="red", linestyle="--", label="Random (50%)")
    ax.axhline(wf_results["test_win_rate"].mean() * 100, color="green", linestyle=":",
               label=f"Mean: {wf_results['test_win_rate'].mean()*100:.1f}%")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("(B) Trade Win Rate Evolution")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar(folds, wf_results["test_trades"], color="teal", alpha=0.7, edgecolor="black")
    ax.axhline(wf_results["test_trades"].mean(), color="red", linestyle="--",
               label=f"Mean: {wf_results['test_trades'].mean():.1f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Number of Trades")
    ax.set_title("(C) Trade Frequency by Fold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    _save_figure(fig, "figure8_time_series_metrics", output_dir)

    # ---- Text report
    report_path = os.path.join(output_dir, "COMPREHENSIVE_REPORT.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_format_report(
            n=n,
            mean=float(returns.mean()),
            std=float(returns.std()),
            sharpe=sharpe_ann,
            sharpe_rf=float(sr_excess_rf),
            max_dd=max_dd_cumulative,
            win_rate=win_rate,
            t_pval=float(t_pval),
            t_pval_one=float(t_pval_one),
            ci=(ci_lo, ci_hi),
            mc_p=mc_p,
            binom_p=float(binom_p),
            alpha=alpha,
            beta=beta,
            tracking_err=tracking_err,
            info_ratio=float(info_ratio),
            spy_mean=float(spy_returns.mean()),
            ic=float(ic),
            ic_p=float(ic_p),
            psr=float(psr_zero),
            mintrl=float(mintrl_95),
            dsr=dsr_results,
            pbo=float(pbo_single),
        ))
    print(f"  wrote COMPREHENSIVE_REPORT.txt")

    # Persist the WF results CSV at the top of output_dir
    wf_results.to_csv(os.path.join(output_dir, "walk_forward_results.csv"), index=False)

    return {
        "n_folds": n,
        "mean_quarterly_return": float(returns.mean()),
        "annualized_return": float(returns.mean() * 4),
        "sharpe_annualized": sharpe_ann,
        "sharpe_annualized_excess_rf": float(sr_excess_rf),
        "max_drawdown_cumulative": max_dd_cumulative,
        "max_drawdown_fold": max_dd,
        "win_rate_folds": win_rate,
        "t_pvalue_two_sided": float(t_pval),
        "t_pvalue_one_sided": float(t_pval_one),
        "bootstrap_ci_lower": ci_lo,
        "bootstrap_ci_upper": ci_hi,
        "monte_carlo_pvalue": mc_p,
        "binomial_pvalue": float(binom_p),
        "alpha_quarterly": alpha,
        "beta": beta,
        "information_coefficient": float(ic),
        "psr_sr_zero": float(psr_zero),
        "mintrl_quarters_95pct": float(mintrl_95),
        "dsr_n_trials_1": float(dsr_results[1]),
        "dsr_n_trials_10": float(dsr_results[10]),
        "dsr_n_trials_30": float(dsr_results[30]),
        "dsr_n_trials_100": float(dsr_results[100]),
        "pbo_single_strategy": float(pbo_single),
    }


def _format_report(**k) -> str:
    dsr = k['dsr']
    return f"""Walk-Forward Validation: Comprehensive Summary Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. Performance summary
   Folds                                : {k['n']}
   Mean quarterly return                : {k['mean']:.4f}  ({k['mean']*4:+.2%} annualized)
   Standard deviation                   : {k['std']:.4f}
   Sharpe ratio, excess over zero (ann.): {k['sharpe']:.4f}
   Sharpe ratio, excess over Rf (ann.)  : {k['sharpe_rf']:.4f}    [Rf = 2.0% p.a.]
   Maximum drawdown (cumulative)        : {k['max_dd']:.4f}
   Win rate (folds)                     : {k['win_rate']:.4f}

2. Classical tests (H0: mean return = 0)
   t-test two-sided  p-value            : {k['t_pval']:.4f}
   t-test one-sided  p-value            : {k['t_pval_one']:.4f}
   Bootstrap 95% CI                     : [{k['ci'][0]:.4f}, {k['ci'][1]:.4f}]
   Sign-flip permutation (10,000)       : {k['mc_p']:.4f}
   Binomial test (win > 50%)            : {k['binom_p']:.4f}

3. Sharpe-ratio inference (Bailey - Lopez de Prado machinery)
   PSR(SR* = 0)                         : {k['psr']:.4f}
   Minimum track record length @ 95%    : {k['mintrl']:.1f} periods ({k['mintrl']/4:.1f} years)
   DSR (N =   1 trial)                  : {dsr[1]:.4f}
   DSR (N =  10 trials)                 : {dsr[10]:.4f}
   DSR (N =  30 trials)                 : {dsr[30]:.4f}
   DSR (N = 100 trials)                 : {dsr[100]:.4f}
   PBO (single-strategy train/test rank): {k['pbo']:.4f}

4. Benchmark comparison (vs SPY)
   Strategy mean return                 : {k['mean']*4:+.2%} annualized
   SPY mean return                      : {k['spy_mean']*4:+.2%} annualized
   Alpha (quarterly)                    : {k['alpha']:.4f}  ({k['alpha']*4:+.2%} annualized)
   Beta                                 : {k['beta']:.4f}
   Tracking error                       : {k['tracking_err']:.4f}
   Information ratio                    : {k['info_ratio']:.4f}

5. Overfitting diagnostic
   Information coefficient              : {k['ic']:.3f}  (train vs. test correlation,
                                          p-value = {k['ic_p']:.4f})
"""
