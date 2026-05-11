"""Hypothesis generation.

Defines the trading-hypothesis data structure, regime detector, and the five
rule-based hypothesis types used as illustrative generators in the paper
(Section 3.3, Appendix 7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    VOLATILE = "volatile"
    STABLE = "stable"


@dataclass
class TradingHypothesis:
    """Structured trading hypothesis (paper Definition: Trading Hypothesis)."""

    hypothesis_id: str
    timestamp: pd.Timestamp
    symbol: str
    action: str
    reasoning_type: str
    natural_language: str
    confidence: float
    features_used: Dict[str, float]
    target_return: float
    stop_loss: float
    sector: str = ""

    def to_dict(self) -> Dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "action": self.action,
            "reasoning_type": self.reasoning_type,
            "natural_language": self.natural_language,
            "confidence": self.confidence,
            "target_return": self.target_return,
            "stop_loss": self.stop_loss,
            "sector": self.sector,
        }


class RegimeDetector:
    """Coarse four-state market regime classifier."""

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        if len(df) < 20:
            return MarketRegime.STABLE
        vol = (
            df["volatility_20d"].iloc[-1]
            if "volatility_20d" in df.columns
            else df["returns"].tail(20).std()
        )
        trend = (
            df["return_20d"].iloc[-1]
            if "return_20d" in df.columns
            else df["Close"].pct_change(20).iloc[-1]
        )
        if vol > 0.02:
            return MarketRegime.VOLATILE
        if abs(trend) < 0.02:
            return MarketRegime.STABLE
        if trend > 0.05:
            return MarketRegime.TRENDING_UP
        return MarketRegime.TRENDING_DOWN


def _safe_get(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        val = row.get(key, default)
        return float(default) if pd.isna(val) else float(val)
    except (TypeError, ValueError):
        return float(default)


class MasterHypothesisGenerator:
    """Rule-based generator producing one hypothesis per (symbol, date).

    Five strategies are evaluated in order; the first matching strategy wins.
    Threshold values are documented in Appendix 7.2 of the manuscript.
    """

    _FEATURE_DEFAULTS = {
        "price": 0.0,
        "rsi": 50.0,
        "volume_ratio": 1.0,
        "volume_imbalance": 0.0,
        "return_20d": 0.0,
        "volatility_20d": 0.01,
        "trend_strength": 0.0,
        "bb_position": 0.5,
        "price_efficiency": 0.5,
        "candle_size_ratio": 1.0,
        "distance_from_high": 0.5,
        "distance_from_low": 0.5,
        "macd_hist": 0.0,
        "return_5d": 0.0,
        "return_10d": 0.0,
    }

    def __init__(self) -> None:
        self.hypothesis_count = 0
        self.regime_detector = RegimeDetector()

    def generate_hypothesis(
        self, df: pd.DataFrame, symbol: str, sector: str = ""
    ) -> Optional[TradingHypothesis]:
        if len(df) < 60:
            return None
        features = self._extract_features(df)
        if not features:
            return None

        regime = self.regime_detector.detect_regime(df)
        self.hypothesis_count += 1
        hyp_id = f"{symbol}_{self.hypothesis_count}_{datetime.now():%Y%m%d}"

        for strategy in (
            self._institutional_accumulation,
            self._flow_momentum,
            self._mean_reversion,
            self._breakout,
            self._range_bound_value,
        ):
            hyp = strategy(features, symbol, hyp_id, sector, regime)
            if hyp is not None:
                return hyp
        return None

    def _extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        if len(df) < 2:
            return {}
        latest = df.iloc[-1]
        key_to_source = {
            "price": "Close",
            "rsi": "rsi",
            "volume_ratio": "volume_ratio",
            "volume_imbalance": "volume_imbalance",
            "return_20d": "return_20d",
            "volatility_20d": "volatility_20d",
            "trend_strength": "trend_strength",
            "bb_position": "bb_position",
            "price_efficiency": "price_efficiency",
            "candle_size_ratio": "candle_size_ratio",
            "distance_from_high": "distance_from_high",
            "distance_from_low": "distance_from_low",
            "macd_hist": "macd_hist",
            "return_5d": "return_5d",
            "return_10d": "return_10d",
        }
        return {
            name: _safe_get(latest, source, self._FEATURE_DEFAULTS[name])
            for name, source in key_to_source.items()
        }

    @staticmethod
    def _institutional_accumulation(features, symbol, hyp_id, sector, regime):
        if (
            features["volume_imbalance"] > 0.30
            and features["volume_ratio"] > 1.5
            and abs(features["return_20d"]) < 0.10
        ):
            return TradingHypothesis(
                hypothesis_id=hyp_id,
                timestamp=pd.Timestamp.now(),
                symbol=symbol,
                action="buy",
                reasoning_type="institutional_accumulation",
                natural_language=(
                    f"{symbol} ({sector}) shows institutional accumulation: "
                    f"{features['volume_imbalance']:.0%} buy imbalance with "
                    f"{features['volume_ratio']:.1f}x volume at stable price."
                ),
                confidence=0.75,
                features_used=features,
                target_return=0.08,
                stop_loss=0.04,
                sector=sector,
            )
        return None

    @staticmethod
    def _flow_momentum(features, symbol, hyp_id, sector, regime):
        if (
            features["return_20d"] > 0.10
            and features["volume_imbalance"] > 0.20
            and features["price_efficiency"] > 0.50
            and features["rsi"] < 80
        ):
            return TradingHypothesis(
                hypothesis_id=hyp_id,
                timestamp=pd.Timestamp.now(),
                symbol=symbol,
                action="buy",
                reasoning_type="flow_momentum",
                natural_language=(
                    f"{symbol} ({sector}) momentum +{features['return_20d']*100:.1f}% "
                    f"confirmed by {features['volume_imbalance']:.0%} order-flow imbalance."
                ),
                confidence=0.70,
                features_used=features,
                target_return=0.10,
                stop_loss=0.05,
                sector=sector,
            )
        return None

    @staticmethod
    def _mean_reversion(features, symbol, hyp_id, sector, regime):
        if (
            regime == MarketRegime.STABLE
            and features["rsi"] < 35
            and features["bb_position"] < 0.20
            and features["distance_from_low"] < 0.10
        ):
            return TradingHypothesis(
                hypothesis_id=hyp_id,
                timestamp=pd.Timestamp.now(),
                symbol=symbol,
                action="buy",
                reasoning_type="mean_reversion",
                natural_language=(
                    f"{symbol} ({sector}) oversold in stable regime: "
                    f"RSI={features['rsi']:.0f}, near lower Bollinger band."
                ),
                confidence=0.65,
                features_used=features,
                target_return=0.05,
                stop_loss=0.03,
                sector=sector,
            )
        return None

    @staticmethod
    def _breakout(features, symbol, hyp_id, sector, regime):
        if (
            features["distance_from_high"] < 0.03
            and features["volume_ratio"] > 1.8
            and features["macd_hist"] > 0
            and 50 < features["rsi"] < 70
        ):
            return TradingHypothesis(
                hypothesis_id=hyp_id,
                timestamp=pd.Timestamp.now(),
                symbol=symbol,
                action="buy",
                reasoning_type="breakout",
                natural_language=(
                    f"{symbol} ({sector}) breaking out near 52-week high with "
                    f"{features['volume_ratio']:.1f}x volume; MACD positive."
                ),
                confidence=0.68,
                features_used=features,
                target_return=0.07,
                stop_loss=0.04,
                sector=sector,
            )
        return None

    @staticmethod
    def _range_bound_value(features, symbol, hyp_id, sector, regime):
        if (
            regime == MarketRegime.STABLE
            and features["rsi"] < 50
            and abs(features["return_20d"]) < 0.05
        ):
            return TradingHypothesis(
                hypothesis_id=hyp_id,
                timestamp=pd.Timestamp.now(),
                symbol=symbol,
                action="buy",
                reasoning_type="range_bound_value",
                natural_language=(
                    f"{symbol} ({sector}) range-bound, RSI={features['rsi']:.0f}; "
                    f"low-volatility accumulation candidate."
                ),
                confidence=0.60,
                features_used=features,
                target_return=0.05,
                stop_loss=0.03,
                sector=sector,
            )
        return None
