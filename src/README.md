
# Hypothesis-Driven Trading: Publication Package

This directory contains all figures, tables, and statistical analyses for the manuscript:

**"Hypothesis-Driven Trading: A Walk-Forward Validation Framework for Market Microstructure Signals"**

## Contents

### Tables
1. `table1_summary_statistics` - Overall performance metrics
2. `table2_regime_performance` - Performance by market regime
3. `table3_statistical_tests` - Statistical significance tests
4. `table4_benchmark_comparison` - Comparison to SPY benchmark
5. `table5_risk_metrics` - Comprehensive risk-adjusted metrics
6. `table6_fold_details` - Fold-by-fold detailed results
7. `table7_hypothesis_performance` - Performance by hypothesis category
8. `table8_literature_comparison` - Comparison to published strategies

### Figures
1. `figure1_main_results` - Four-panel main results (returns, cumulative, win rate, Sharpe)
2. `figure2_statistical_analysis` - Statistical tests (bootstrap, permutation, Q-Q plot)
3. `figure3_regime_comparison` - Performance by market regime
4. `figure4_benchmark_comparison` - Strategy vs SPY
5. `figure5_drawdown_analysis` - Drawdown profile
6. `figure6_train_vs_test` - Training vs testing correlation
7. `figure7_return_distribution` - Distribution characteristics
8. `figure8_time_series_metrics` - Time evolution of metrics

### Reports
- `COMPREHENSIVE_REPORT.txt` - Complete analysis summary

## File Formats
- Tables: Available in both CSV and LaTeX format
- Figures: Available in both PNG (300 DPI) and PDF (vector) format
- All files are publication-ready

## Usage for Manuscript

### LaTeX Integration
```latex
% Include tables
\input{table1_summary_statistics.tex}

% Include figures
\begin{figure}
  \centering
  \includegraphics[width=\textwidth]{figure1_main_results.pdf}
  \caption{Main Walk-Forward Results}
  \label{fig:main}
\end{figure}
```

### Citation
If using this framework, please cite:
[Your name], [Year]. "Hypothesis-Driven Trading: A Walk-Forward Validation Framework."
[Journal], [Volume]([Issue]), [Pages].

## Reproducibility
All results generated from walk-forward validation with:
- Train window: 252 days (1 year)
- Test window: 63 days (1 quarter)
- No lookahead bias
- Realistic transaction costs (commission + slippage)
- Position limits and risk management

## Contact
[Your name]
[Your email]
[Your institution]
