"""Reproduce the walk-forward validation results reported in the manuscript.

Pipeline
--------
1. Download (or load cached) daily OHLCV bars for 100 US equities + SPY,
   2015-01-01 -> 2024-11-01.
2. Engineer the 54-feature vector for each security.
3. Run walk-forward validation (252-day train / 63-day test, non-overlapping).
4. Generate publication tables, figures, and a summary report.

Usage
-----
    python main.py
    python main.py --start 2020-01-01     # 14-fold sub-sample
    python main.py --force-refresh        # ignore the data cache
    python main.py --output ./outputs_v2  # alternate output directory

Outputs land in ``--output`` (default: ``./outputs``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from hdt import analysis
from hdt.config import (
    ALL_SYMBOLS,
    BENCHMARK,
    DATA_CACHE_DIR,
    END_DATE,
    INITIAL_CAPITAL,
    OUTPUT_DIR,
    RANDOM_SEED,
    START_DATE,
    TEST_WINDOW_DAYS,
    TRAIN_WINDOW_DAYS,
)
from hdt.data_loader import DataLoader
from hdt.features import add_features_to_all
from hdt.hypothesis import MasterHypothesisGenerator
from hdt.validation import WalkForwardValidator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=START_DATE, help=f"Start date (default {START_DATE})")
    p.add_argument("--end", default=END_DATE, help=f"End date (default {END_DATE})")
    p.add_argument("--train-window", type=int, default=TRAIN_WINDOW_DAYS,
                   help=f"Training window in days (default {TRAIN_WINDOW_DAYS})")
    p.add_argument("--test-window", type=int, default=TEST_WINDOW_DAYS,
                   help=f"Test window in days (default {TEST_WINDOW_DAYS})")
    p.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                   help=f"Starting capital per fold (default {INITIAL_CAPITAL})")
    p.add_argument("--cache-dir", default=DATA_CACHE_DIR, help="Data cache directory")
    p.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    p.add_argument("--force-refresh", action="store_true", help="Re-download data")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    # 1. Download / load data
    loader = DataLoader(cache_dir=args.cache_dir)
    raw = loader.download_data(
        symbols=ALL_SYMBOLS + [BENCHMARK],
        start_date=args.start,
        end_date=args.end,
        force_refresh=args.force_refresh,
    )
    print(f"Loaded {len(raw)} symbols, "
          f"{len(raw[BENCHMARK])} trading days ({raw[BENCHMARK].index[0].date()} -> "
          f"{raw[BENCHMARK].index[-1].date()})")

    # 2. Feature engineering
    print("Engineering features...")
    enhanced = add_features_to_all(raw)
    print(f"Features ready for {len(enhanced)} symbols "
          f"({len(enhanced[BENCHMARK].columns)} columns each)")

    # 3. Walk-forward validation
    generator = MasterHypothesisGenerator()
    validator = WalkForwardValidator(
        train_window_days=args.train_window,
        test_window_days=args.test_window,
        initial_capital=args.capital,
    )
    wf_results = validator.validate(enhanced, generator, seed=args.seed, verbose=True)
    wf_results.to_csv(os.path.join(args.output, "walk_forward_results.csv"), index=False)

    # 4. Publication outputs
    print("Generating publication outputs...")
    summary = analysis.generate_publication_outputs(
        wf_results, enhanced, output_dir=args.output, rng_seed=args.seed
    )

    summary_path = os.path.join(args.output, "summary_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. Outputs in {args.output}/")
    print("Headline metrics:")
    for k, v in summary.items():
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
