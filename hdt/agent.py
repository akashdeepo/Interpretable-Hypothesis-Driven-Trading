"""Reinforcement-learning agent (epsilon-greedy bandit over hypothesis types).

The agent maintains per-type counts, wins, and average returns, and decides
whether to execute an incoming hypothesis based on its learned win rate. See
paper Section 3.4 (Definitions: Agent State, epsilon-Greedy Policy).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd

from .hypothesis import TradingHypothesis


class RLAgent:
    def __init__(self, epsilon: float = 0.5, rng: np.random.Generator | None = None) -> None:
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()
        self.hypothesis_performance: Dict[str, Dict] = defaultdict(
            lambda: {
                "count": 0,
                "wins": 0,
                "total_return": 0.0,
                "returns": [],
                "avg_return": 0.0,
                "win_rate": 0.0,
            }
        )
        self.sector_performance: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "wins": 0, "total_return": 0.0}
        )

    def should_execute(self, hypothesis: TradingHypothesis) -> bool:
        """Epsilon-greedy decision."""
        if self.rng.random() < self.epsilon:
            return True
        stats = self.hypothesis_performance[hypothesis.reasoning_type]
        if stats["count"] < 5:
            return hypothesis.confidence > 0.5
        threshold = 0.45 + (1.0 - hypothesis.confidence) * 0.10
        return stats["win_rate"] > threshold

    def record_outcome(self, hypothesis: TradingHypothesis, actual_return: float) -> None:
        stats = self.hypothesis_performance[hypothesis.reasoning_type]
        stats["count"] += 1
        stats["total_return"] += actual_return
        stats["returns"].append(actual_return)
        if actual_return > 0:
            stats["wins"] += 1
        stats["avg_return"] = stats["total_return"] / stats["count"]
        stats["win_rate"] = stats["wins"] / stats["count"]

        if hypothesis.sector:
            ss = self.sector_performance[hypothesis.sector]
            ss["count"] += 1
            ss["total_return"] += actual_return
            if actual_return > 0:
                ss["wins"] += 1

    def get_performance_summary(self) -> pd.DataFrame:
        rows = []
        for htype, s in self.hypothesis_performance.items():
            if s["count"] == 0:
                continue
            sharpe = (
                s["avg_return"] / (np.std(s["returns"]) + 1e-6)
                if len(s["returns"]) > 1
                else 0.0
            )
            rows.append(
                {
                    "hypothesis_type": htype,
                    "count": s["count"],
                    "win_rate": s["win_rate"],
                    "avg_return": s["avg_return"],
                    "total_return": s["total_return"],
                    "sharpe": sharpe,
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("win_rate", ascending=False)
