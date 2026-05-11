# Code: how to reproduce the manuscript's walk-forward results

This directory implements the framework from *"Interpretable Hypothesis-Driven
Trading: A Rigorous Walk-Forward Validation Framework for Market Microstructure
Signals"* (Deep, Deep, & Lamptey, 2025) as a small Python package.

## Layout

```
.
├── hdt/                     # the framework
│   ├── config.py            # universe, dates, walk-forward parameters
│   ├── data_loader.py       # yfinance downloader + on-disk cache
│   ├── features.py          # 54-feature engineering pipeline
│   ├── hypothesis.py        # trading-hypothesis structure + 5 rule-based generators
│   ├── agent.py             # epsilon-greedy RL agent
│   ├── backtester.py        # event-driven backtester (commissions, slippage, limits)
│   ├── validation.py        # rolling-window walk-forward loop
│   └── analysis.py          # statistical tests, tables, figures, report
├── main.py                  # entry point
├── requirements.txt
├── src/                     # archived figures/tables that ship with the paper
├── main.tex                 # manuscript source
├── references.bib
└── Final_HMM_RL_interpretable_ML (1).ipynb   # original notebook (kept for reference)
```

## Reproducing the results

```bash
python -m pip install -r requirements.txt
python main.py                            # full sample: 2015-01-01 -> 2024-11-01 (34 folds)
python main.py --start 2020-01-01         # short sample: 14 folds (notebook's first run)
python main.py --output ./outputs_v2      # alternate output directory
python main.py --force-refresh            # re-download data instead of using the cache
```

A complete run (100 equities + SPY, 34 folds) takes roughly 30-50 minutes on a
modern laptop. The first run downloads ~50 MB of daily bars; subsequent runs
re-use the cache in `./data_cache/`.

## Outputs

`main.py` writes everything to `./outputs/` (configurable):

| File                                 | Content |
|--------------------------------------|---------|
| `walk_forward_results.csv`           | fold-by-fold metrics |
| `summary_metrics.json`               | aggregate metrics, p-values, beta/alpha |
| `table1_summary_statistics.{csv,tex}`| performance summary |
| `table2_regime_performance.{csv,tex}`| performance by regime |
| `table3_statistical_tests.{csv,tex}` | t-test, bootstrap, permutation, Shapiro-Wilk, binomial |
| `table4_benchmark_comparison.{csv,tex}` | strategy vs SPY |
| `table5_risk_metrics.{csv,tex}`      | VaR, CVaR, Sortino, Calmar, drawdowns |
| `table6_fold_details.{csv,tex}`      | every fold's dates, return, trades, regime |
| `figure1_main_results.{png,pdf}`     | four-panel main results |
| `figure2_statistical_analysis.{png,pdf}` | bootstrap / MC permutation / Q-Q plot |
| `figure3_regime_comparison.{png,pdf}`| boxplot + bars by regime |
| `figure4_benchmark_comparison.{png,pdf}` | strategy vs SPY |
| `figure5_drawdown_analysis.{png,pdf}`| cumulative returns + drawdown |
| `figure6_train_vs_test.{png,pdf}`    | overfitting diagnostic |
| `figure7_return_distribution.{png,pdf}` | distribution / CDF / violin / ACF |
| `figure8_time_series_metrics.{png,pdf}` | returns, win-rate, trade-count over time |
| `COMPREHENSIVE_REPORT.txt`           | plain-text summary |

The files under the existing `src/` directory are the artefacts referenced in
the manuscript (`\graphicspath{{./src/}}`); they are kept verbatim. Re-running
the pipeline produces an independent copy under `outputs/` for verification.

## Determinism notes

Reproducibility is not bit-exact:

- `yfinance` occasionally back-revises historical adjusted prices; small price
  differences propagate into derived features and trade outcomes.
- The RL agent uses a fold-seeded `numpy.random.Generator`. The bootstrap and
  Monte Carlo permutation tests are also seeded.

With the default seed, headline metrics (mean return, Sharpe, beta, max
drawdown, p-values) match the manuscript to within rounding for the 34-fold
sample.

## Original notebook

The original development notebook is preserved as
`Final_HMM_RL_interpretable_ML (1).ipynb`. The package in `hdt/` is a clean
re-implementation; do not rely on the notebook for new analysis.
