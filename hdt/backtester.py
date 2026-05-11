"""Event-driven backtester.

Signals are generated using information up to and including day t (close);
orders execute at day (t + 1) open with slippage and a fixed commission.
Implements the position-sizing, sector-exposure and exit rules described in
paper Section 3.6.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .agent import RLAgent
from .config import (
    COMMISSION_PER_TRADE,
    MAX_HOLDING_DAYS,
    MAX_POSITIONS,
    MAX_POSITION_PCT,
    MAX_SECTOR_EXPOSURE,
    SLIPPAGE_BPS,
    UNIVERSE,
)
from .hypothesis import TradingHypothesis


def _sector_of(symbol: str) -> str:
    for sector, stocks in UNIVERSE.items():
        if symbol in stocks:
            return sector
    return "unknown"


class ProductionBacktester:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_per_trade: float = COMMISSION_PER_TRADE,
        slippage_bps: float = SLIPPAGE_BPS,
        max_positions: int = MAX_POSITIONS,
        max_position_pct: float = MAX_POSITION_PCT,
        max_sector_exposure: float = MAX_SECTOR_EXPOSURE,
        max_holding_days: int = MAX_HOLDING_DAYS,
    ) -> None:
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trade_history: list = []
        self.portfolio_value_history: list = []

        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.max_sector_exposure = max_sector_exposure
        self.max_holding_days = max_holding_days

    def run_backtest(
        self,
        market_data: Dict[str, pd.DataFrame],
        generator,
        rl_agent: RLAgent,
        start_idx: int = 60,
        check_frequency: int = 1,
        verbose: bool = False,
    ) -> Dict[str, int]:
        symbols = [s for s in market_data.keys() if s != "SPY"]
        total_days = len(market_data["SPY"])

        trades_attempted = 0
        trades_executed = 0

        it = tqdm(
            range(start_idx, total_days - 1, check_frequency),
            desc="Backtesting",
            disable=not verbose,
        )
        for day_idx in it:
            current_date = market_data["SPY"].index[day_idx]
            lookback = {sym: df.iloc[: day_idx + 1] for sym, df in market_data.items()}

            self._check_positions(market_data, day_idx, current_date, rl_agent)

            hypotheses = []
            for symbol in symbols:
                hyp = generator.generate_hypothesis(
                    lookback[symbol], symbol, _sector_of(symbol)
                )
                if hyp is not None:
                    hypotheses.append(hyp)

            if len(self.positions) < self.max_positions and hypotheses:
                for hyp in hypotheses:
                    if hyp.symbol in self.positions:
                        continue
                    if len(self.positions) >= self.max_positions:
                        break
                    if not self._check_sector_exposure(hyp):
                        continue
                    trades_attempted += 1
                    if rl_agent.should_execute(hyp) and hyp.action == "buy":
                        if self._open_position(hyp, market_data, day_idx + 1, current_date):
                            trades_executed += 1

            self.portfolio_value_history.append(
                {
                    "date": current_date,
                    "value": self._calculate_portfolio_value(lookback),
                    "cash": self.capital,
                    "positions": len(self.positions),
                }
            )

        self._close_all_positions(market_data, total_days - 1, rl_agent)
        return {"trades_attempted": trades_attempted, "trades_executed": trades_executed}

    # ---------------------------------------------------------------- helpers

    def _check_sector_exposure(self, hypothesis: TradingHypothesis) -> bool:
        sector = hypothesis.sector
        if not sector:
            return True
        sector_value = sum(
            pos["shares"] * pos["current_price"]
            for sym, pos in self.positions.items()
            if _sector_of(sym) == sector
        )
        total = self._calculate_portfolio_value_simple()
        return (sector_value / total if total > 0 else 0) < self.max_sector_exposure

    def _open_position(
        self,
        hypothesis: TradingHypothesis,
        market_data: Dict[str, pd.DataFrame],
        execution_idx: int,
        signal_date: pd.Timestamp,
    ) -> bool:
        symbol = hypothesis.symbol
        if execution_idx >= len(market_data[symbol]):
            return False
        execution_price = market_data[symbol]["Open"].iloc[execution_idx]
        execution_date = market_data[symbol].index[execution_idx]

        slippage = execution_price * (self.slippage_bps / 10_000.0)
        fill_price = execution_price + slippage

        max_value = self.capital * self.max_position_pct
        shares = int(max_value / fill_price)
        if shares == 0:
            return False

        gross_cost = shares * fill_price
        total_cost = gross_cost + self.commission_per_trade
        if total_cost > self.capital:
            return False

        self.capital -= total_cost
        self.positions[symbol] = {
            "shares": shares,
            "entry_price": fill_price,
            "entry_date": execution_date,
            "entry_idx": execution_idx,
            "signal_date": signal_date,
            "hypothesis": hypothesis,
            "cost": gross_cost,
            "entry_commission": self.commission_per_trade,
            "current_price": fill_price,
        }
        return True

    def _check_positions(
        self,
        market_data: Dict[str, pd.DataFrame],
        day_idx: int,
        current_date: pd.Timestamp,
        rl_agent: RLAgent,
    ) -> None:
        to_close = []
        for symbol, pos in self.positions.items():
            price = market_data[symbol]["Close"].iloc[day_idx]
            pos["current_price"] = price
            ret = (price - pos["entry_price"]) / pos["entry_price"]
            hyp = pos["hypothesis"]
            held = day_idx - pos["entry_idx"]

            if ret >= hyp.target_return:
                reason = "TARGET_HIT"
            elif ret <= -hyp.stop_loss:
                reason = "STOP_LOSS"
            elif held > self.max_holding_days:
                reason = "TIME_LIMIT"
            else:
                continue

            if day_idx + 1 < len(market_data[symbol]):
                exit_price = market_data[symbol]["Open"].iloc[day_idx + 1]
                exit_date = market_data[symbol].index[day_idx + 1]
            else:
                exit_price = price
                exit_date = current_date
            to_close.append((symbol, exit_price, exit_date, reason))

        for symbol, exit_price, exit_date, reason in to_close:
            self._close_position(symbol, exit_price, exit_date, reason, rl_agent)

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: pd.Timestamp,
        reason: str,
        rl_agent: RLAgent,
    ) -> None:
        pos = self.positions[symbol]
        shares = pos["shares"]

        slippage = exit_price * (self.slippage_bps / 10_000.0)
        fill_price = exit_price - slippage

        gross = shares * fill_price
        net = gross - self.commission_per_trade
        self.capital += net

        total_cost = pos["cost"] + pos["entry_commission"]
        profit = net - total_cost
        ret = profit / total_cost

        self.trade_history.append(
            {
                "symbol": symbol,
                "entry_date": pos["entry_date"],
                "exit_date": exit_date,
                "entry_price": pos["entry_price"],
                "exit_price": fill_price,
                "shares": shares,
                "return": ret,
                "profit": profit,
                "reason": reason,
                "hypothesis_type": pos["hypothesis"].reasoning_type,
                "hypothesis": pos["hypothesis"],
                "days_held": (exit_date - pos["entry_date"]).days,
                "total_costs": pos["entry_commission"]
                + self.commission_per_trade
                + (slippage * shares * 2),
                "sector": pos["hypothesis"].sector,
            }
        )
        rl_agent.record_outcome(pos["hypothesis"], ret)
        del self.positions[symbol]

    def _close_all_positions(
        self, market_data: Dict[str, pd.DataFrame], day_idx: int, rl_agent: RLAgent
    ) -> None:
        for symbol in list(self.positions.keys()):
            price = market_data[symbol]["Close"].iloc[day_idx]
            date = market_data[symbol].index[day_idx]
            self._close_position(symbol, price, date, "BACKTEST_END", rl_agent)

    def _calculate_portfolio_value(self, market_data: Dict[str, pd.DataFrame]) -> float:
        value = self.capital
        for symbol, pos in self.positions.items():
            value += pos["shares"] * market_data[symbol]["Close"].iloc[-1]
        return value

    def _calculate_portfolio_value_simple(self) -> float:
        return self.capital + sum(
            pos["shares"] * pos["current_price"] for pos in self.positions.values()
        )

    def get_performance_summary(self) -> Dict[str, float]:
        empty = {
            "total_return": 0.0,
            "final_capital": self.capital,
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_return_per_trade": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_profit": 0.0,
            "total_costs": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_days_held": 0.0,
        }
        if not self.trade_history:
            return empty

        df = pd.DataFrame(self.trade_history)
        returns = df["return"].values
        win_rate = float((returns > 0).sum()) / len(returns)

        if len(returns) > 1 and df["days_held"].mean() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 / df["days_held"].mean())
        else:
            sharpe = 0.0

        if self.portfolio_value_history:
            pv = pd.DataFrame(self.portfolio_value_history)["value"]
            dd = (pv - pv.cummax()) / pv.cummax()
            max_drawdown = float(dd.min())
        else:
            max_drawdown = 0.0

        total_return = (self.capital - self.initial_capital) / self.initial_capital
        return {
            "total_return": float(total_return),
            "final_capital": float(self.capital),
            "num_trades": int(len(df)),
            "win_rate": win_rate,
            "avg_return_per_trade": float(returns.mean()),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_drawdown,
            "total_profit": float(df["profit"].sum()),
            "total_costs": float(df["total_costs"].sum()),
            "best_trade": float(returns.max()),
            "worst_trade": float(returns.min()),
            "avg_days_held": float(df["days_held"].mean()),
        }
