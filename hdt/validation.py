"""Walk-forward validation.

Slices the time series into rolling (train, test) folds, trains an RL agent on
each training window, and evaluates it on the immediately-following test
window. See paper Section 3.5 (Definition: Walk-Forward Partition).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .agent import RLAgent
from .backtester import ProductionBacktester
from .config import (
    EPSILON_TEST,
    EPSILON_TRAIN,
    INITIAL_CAPITAL,
    TEST_WINDOW_DAYS,
    TRAIN_WINDOW_DAYS,
)


class WalkForwardValidator:
    def __init__(
        self,
        train_window_days: int = TRAIN_WINDOW_DAYS,
        test_window_days: int = TEST_WINDOW_DAYS,
        initial_capital: float = INITIAL_CAPITAL,
        epsilon_train: float = EPSILON_TRAIN,
        epsilon_test: float = EPSILON_TEST,
    ) -> None:
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.initial_capital = initial_capital
        self.epsilon_train = epsilon_train
        self.epsilon_test = epsilon_test

    def validate(
        self,
        market_data: Dict[str, pd.DataFrame],
        generator,
        seed: int | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        total_days = len(market_data["SPY"])
        idx = self.train_window + 60
        results = []
        fold = 0

        if verbose:
            print("Walk-forward validation")
            print(f"  train window: {self.train_window} days")
            print(f"  test  window: {self.test_window} days")
            print(f"  total days  : {total_days}")

        while idx + self.test_window < total_days:
            fold += 1
            train_start, train_end = idx - self.train_window, idx
            test_start, test_end = idx, min(idx + self.test_window, total_days)

            train_data = {s: df.iloc[train_start:train_end] for s, df in market_data.items()}
            test_data = {s: df.iloc[test_start:test_end] for s, df in market_data.items()}

            fold_seed = None if seed is None else seed + fold
            rng = np.random.default_rng(fold_seed)

            rl_agent = RLAgent(epsilon=self.epsilon_train, rng=rng)
            train_bt = ProductionBacktester(initial_capital=self.initial_capital)
            train_bt.run_backtest(train_data, generator, rl_agent, start_idx=60, verbose=False)
            train_perf = train_bt.get_performance_summary()

            rl_agent.epsilon = self.epsilon_test
            test_bt = ProductionBacktester(initial_capital=self.initial_capital)
            test_bt.run_backtest(test_data, generator, rl_agent, start_idx=1, verbose=False)
            test_perf = test_bt.get_performance_summary()

            row = {
                "fold": fold,
                "train_start": market_data["SPY"].index[train_start],
                "train_end": market_data["SPY"].index[train_end - 1],
                "test_start": market_data["SPY"].index[test_start],
                "test_end": market_data["SPY"].index[test_end - 1],
                "train_return": train_perf["total_return"],
                "train_trades": train_perf["num_trades"],
                "test_return": test_perf["total_return"],
                "test_trades": test_perf["num_trades"],
                "test_win_rate": test_perf["win_rate"],
                "test_sharpe": test_perf["sharpe_ratio"],
                "test_max_dd": test_perf["max_drawdown"],
            }
            results.append(row)

            if verbose:
                print(
                    f"  fold {fold:2d}  "
                    f"{row['test_start'].date()}..{row['test_end'].date()}  "
                    f"test_return={row['test_return']:+.4f}  "
                    f"trades={row['test_trades']:3d}  "
                    f"win={row['test_win_rate']:.2f}  "
                    f"sharpe={row['test_sharpe']:+.2f}"
                )

            idx += self.test_window

        return pd.DataFrame(results)
