"""Feature engineering.

Constructs the 54-feature vector used by the hypothesis generators, organised
into four groups: technical indicators, momentum/volatility, volume and
microstructure, and regime indicators (paper Section 3.2 and Appendix 7.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` augmented with engineered feature columns."""
    df = df.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    # Technical indicators
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()
    df["ema_12"] = close.ewm(span=12).mean()
    df["ema_26"] = close.ewm(span=26).mean()

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Momentum and volatility
    for n in (1, 5, 10, 20, 60):
        df[f"return_{n}d"] = close.pct_change(n)
    df["volatility_5d"] = df["returns"].rolling(5).std()
    df["volatility_20d"] = df["returns"].rolling(20).std()
    df["volatility_60d"] = df["returns"].rolling(60).std()

    # Volume
    df["volume_sma_20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_sma_20"]
    df["vwap"] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
    df["vwap_distance"] = (close - df["vwap"]) / df["vwap"]
    df["obv"] = (np.sign(df["returns"]) * volume).fillna(0).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=20).mean()

    # Microstructure
    df["price_impact"] = df["returns"].abs() / (df["volume_ratio"] + 1e-6)
    up = df["Close"] > df["Open"]
    down = df["Close"] < df["Open"]
    df["volume_up"] = volume.where(up, 0).rolling(5).sum()
    df["volume_down"] = volume.where(down, 0).rolling(5).sum()
    df["volume_imbalance"] = (
        (df["volume_up"] - df["volume_down"])
        / (df["volume_up"] + df["volume_down"] + 1e-6)
    )
    df["hl_spread"] = (high - low) / close
    df["hl_spread_ma"] = df["hl_spread"].rolling(20).mean()
    df["candle_size"] = (high - low) / close
    df["candle_size_ratio"] = df["candle_size"] / df["candle_size"].rolling(20).mean()
    abs_ret = df["returns"].abs().rolling(10).sum()
    net_ret = df["returns"].rolling(10).sum().abs()
    df["price_efficiency"] = net_ret / (abs_ret + 1e-6)

    # Regime indicators
    df["trend_strength"] = (close - df["sma_50"]) / df["sma_50"]
    vol_ma = df["volatility_20d"].rolling(60).mean()
    df["vol_regime"] = df["volatility_20d"] / vol_ma
    df["high_52w"] = close.rolling(252).max()
    df["low_52w"] = close.rolling(252).min()
    df["distance_from_high"] = (df["high_52w"] - close) / df["high_52w"]
    df["distance_from_low"] = (close - df["low_52w"]) / close

    return df


def add_features_to_all(market_data: dict) -> dict:
    """Apply ``add_all_features`` to every DataFrame in ``market_data``.

    Frames aligned to the benchmark calendar carry all-NaN rows on dates the
    symbol did not trade (e.g. KHC before 2015-07-06). Indicators must see
    only traded bars — rolling/ewm windows spanning the listing boundary would
    otherwise ingest fabricated zeros — so features are computed on the listed
    rows and reindexed back to the full calendar.
    """
    out = {}
    for symbol, df in market_data.items():
        try:
            listed = df.dropna(subset=["Close"])
            feats = add_all_features(listed)
            out[symbol] = feats if len(listed) == len(df) else feats.reindex(df.index)
        except Exception as exc:  # noqa: BLE001
            print(f"  feature engineering failed for {symbol}: {exc}")
    return out
