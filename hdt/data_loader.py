"""Data acquisition: yfinance downloader with on-disk caching."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


class DataLoader:
    """Download daily OHLCV bars and cache them as a pickle.

    The cache key is the (start_date, end_date) pair, so re-running the
    pipeline with the same window reads from disk rather than the API.
    """

    def __init__(self, cache_dir: str = "./data_cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def download_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Return a dict ``{symbol: DataFrame}`` for the requested window."""
        cache_file = os.path.join(
            self.cache_dir, f"data_{start_date}_{end_date}.pkl"
        )
        if not force_refresh and os.path.exists(cache_file):
            print(f"Loading cached data: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        print(f"Downloading {len(symbols)} symbols ({start_date} -> {end_date})")
        market_data: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        for symbol in tqdm(symbols, desc="Downloading"):
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                )
                if len(df) < 50:
                    failed.append(symbol)
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df["returns"] = df["Close"].pct_change()
                df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
                df["volume_ma"] = df["Volume"].rolling(20).mean()
                df["volume_ratio"] = df["Volume"] / df["volume_ma"]
                df["high_low_ratio"] = (df["High"] - df["Low"]) / df["Close"]

                market_data[symbol] = df
            except Exception as exc:  # noqa: BLE001
                failed.append(symbol)
                print(f"  download failed for {symbol}: {exc}")

        print(f"Successfully downloaded {len(market_data)}/{len(symbols)} symbols")
        if failed:
            print(f"  failed: {failed}")

        with open(cache_file, "wb") as f:
            pickle.dump(market_data, f)
        return market_data
