"""Re-run only the publication-analysis phase on the already-saved WF results.

Useful after fixing a bug in ``hdt/analysis.py`` -- avoids the ~30 minute
walk-forward backtest.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from hdt import analysis
from hdt.config import ALL_SYMBOLS, BENCHMARK, DATA_CACHE_DIR, END_DATE, OUTPUT_DIR, START_DATE
from hdt.data_loader import DataLoader
from hdt.features import add_features_to_all


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wf-csv", default=f"{OUTPUT_DIR}/walk_forward_results.csv")
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    wf = pd.read_csv(args.wf_csv, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    print(f"Loaded {len(wf)} folds from {args.wf_csv}")

    loader = DataLoader(cache_dir=DATA_CACHE_DIR)
    raw = loader.download_data(
        symbols=ALL_SYMBOLS + [BENCHMARK],
        start_date=args.start,
        end_date=args.end,
        force_refresh=False,
    )
    enhanced = add_features_to_all(raw)

    summary = analysis.generate_publication_outputs(
        wf, enhanced, output_dir=args.output, rng_seed=args.seed
    )
    with open(f"{args.output}/summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nHeadline metrics:")
    for k, v in summary.items():
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
