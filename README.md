# Interpretable Hypothesis-Driven Trading

## A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals


This repository contains the official implementation of the framework described in the paper: **"Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals"**.


-----

## 📌 Abstract

Quantitative trading research faces a reproducibility crisis, often characterized by overfitting and lookahead bias. This project implements a rigorous **Walk-Forward Validation (WFV)** framework designed to test trading hypotheses under strict out-of-sample conditions.

Unlike "black-box" deep learning models, this system relies on **interpretable, hypothesis-driven signals** (e.g., Institutional Accumulation, Flow Momentum) derived from market microstructure and daily OHLCV data. A Reinforcement Learning (RL) agent adapts strategy selection based on regime performance, while the validation engine ensures that no future information leaks into the training process.

The framework was tested on 100 US Equities (2015–2024), demonstrating market-neutral characteristics ($\beta \approx 0.06$) and exceptional downside protection during the 2022 bear market.

-----

## 🛠 Key Features

  * **Hypothesis-Driven Engine:** Generates signals based on economic logic (e.g., Volume Imbalance, Price Efficiency) rather than opaque patterns. Every trade includes a natural language explanation.
  * **Rigorous Walk-Forward Validation:** Implements a rolling window approach (Train: 1 Year, Test: 1 Quarter) to prevent lookahead bias.
  * **Reinforcement Learning Agent:** An $\epsilon$-greedy bandit algorithm that learns which hypothesis types perform best in the current market regime.
  * **Market Microstructure Features:** Extracts volume imbalance, effective spread proxies, and order flow toxicity from daily data.
  * **Realistic Simulation:** Accounts for transaction costs, slippage, and position limits.
  * **Publication-Ready Analysis:** Automatically generates statistical tests (Bootstrap, Monte Carlo Permutation), regime analysis, and high-resolution figures.

-----

## 📂 Repository Structure

```text
.
├── src/
│   ├── data_loader.py       # yfinance integration and caching
│   ├── features.py          # 54-factor feature engineering engine
│   ├── hypothesis.py        # Logic for "Institutional Accumulation", "Breakouts", etc.
│   ├── agent.py             # Reinforcement Learning (RL) agent logic
│   ├── backtester.py        # Event-driven backtesting engine
│   └── validation.py        # Walk-Forward Validation loop
├── publication_outputs/     # Generated tables (LaTeX/CSV) and Figures (PNG/PDF)
├── main.py                  # Entry point to run the full validation pipeline
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.8 or higher
  * pip

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/akashdeepo/hypothesis-driven-trading.git
    cd hypothesis-driven-trading
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Framework

To run the full pipeline (Data Download $\to$ Feature Engineering $\to$ WFV $\to$ Analysis), execute:

```bash
python main.py
```

*Note: The initial run will download historical data for 100+ tickers, which may take several minutes. Subsequent runs will use the local cache.*

-----

## 📊 Methodology Overview

The core of this repository is the **Walk-Forward Validation** protocol, designed to simulate the experience of a trader operating in real-time without the benefit of hindsight.

1.  **Data Partitioning:** The timeline (2015-2024) is sliced into overlapping windows.
2.  **Training Phase ($W=252$ days):** The RL agent explores hypothesis types and builds a probability distribution of success.
3.  **Testing Phase ($H=63$ days):** The agent freezes its learning parameters and executes strictly out-of-sample.
4.  **Rolling:** The window shifts forward by $H$, and the process repeats 34 times across the dataset.

### Hypothesis Types Implemented

1.  **Institutional Accumulation:** High volume imbalance + stable price.
2.  **Flow Momentum:** Price trend confirmed by order flow.
3.  **Mean Reversion:** Oversold conditions in stable regimes.
4.  **Breakout:** High volume moves near 52-week highs.
5.  **Range-Bound Value:** Accumulation in low-volatility environments.

-----

## 📈 Results

The framework automatically generates a suite of analysis files in the `publication_outputs/` directory. Key findings from the 2015-2024 validation period include:

  * **Annualized Return:** \~0.55% (Unlevered)
  * **Sharpe Ratio:** 0.33
  * **Max Drawdown:** -2.76% (vs -23.8% SPY)
  * **Beta:** 0.058 (Market Neutral)

*While returns are modest, the low correlation and drawdown highlight the effectiveness of microstructure signals for risk management.*

-----

## 📚 Citation

If you use this code or framework in your research, please cite the paper:

```bibtex
@article{deep2025interpretable,
  title={Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals},
  author={Deep, Gagan and Deep, Akash and Lamptey, William},
  journal={Working Paper},
  year={2025},
  institution={Texas Tech University}
}
```

-----

## ⚠️ Disclaimer

This code is provided for educational and research purposes only. It is **not** financial advice. Algorithmic trading involves significant risk of loss. The authors assume no responsibility for any financial losses incurred through the use of this software.

-----

## 📝 License

This project is licensed under the MIT License.

-----

### Acknowledgments

  * **Yahoo Finance** for providing public market data.
  * **Texas Tech University** Department of Mathematics & Statistics.

-----
