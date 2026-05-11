"""
Configuration: universe of securities, sample period, and global parameters.

This is the single source of truth for the experimental setup described in the
paper (100 US equities across 10 GICS sectors, 2015-01-02 -- 2024-10-31).
"""

# ---------------------------------------------------------------------------
# Universe: 100 US equities, 10 per sector (matches paper Section 4.1)
# ---------------------------------------------------------------------------
UNIVERSE = {
    'mega_cap_tech':    ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'AMD', 'INTC', 'CRM'],
    'tech_mid':         ['ADBE', 'NFLX', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC'],
    'financials':       ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'BLK', 'SCHW'],
    'healthcare':       ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'LLY', 'AMGN', 'GILD'],
    'consumer_disc':    ['HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'CMG', 'MAR', 'YUM'],
    'consumer_staples': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'MDLZ', 'CL', 'KMB', 'GIS', 'KHC'],
    'industrials':      ['BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'MMM', 'DE', 'EMR'],
    'energy':           ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'MPC', 'VLO', 'OXY', 'HAL'],
    'materials':        ['LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM', 'DD'],
    'utilities':        ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED'],
}

ALL_SYMBOLS = sorted({s for stocks in UNIVERSE.values() for s in stocks})
BENCHMARK = 'SPY'

# ---------------------------------------------------------------------------
# Sample period
# ---------------------------------------------------------------------------
# Original notebook (cell 0) used 2020-01-01..2024-11-01 (14 folds).
# Extended sample (cell 5) used 2015-01-01..2024-11-01 (34 folds) -- this is
# the version reported in the paper.
START_DATE = '2015-01-01'
END_DATE   = '2024-11-01'

# ---------------------------------------------------------------------------
# Walk-forward parameters (paper Section 3.5)
# ---------------------------------------------------------------------------
TRAIN_WINDOW_DAYS = 252   # W: ~1 year
TEST_WINDOW_DAYS  = 63    # H: ~1 quarter
# step Delta = TEST_WINDOW_DAYS  (non-overlapping test windows)

# ---------------------------------------------------------------------------
# Backtester parameters (paper Section 3.6)
# ---------------------------------------------------------------------------
INITIAL_CAPITAL      = 100_000.0
COMMISSION_PER_TRADE = 1.0     # USD
SLIPPAGE_BPS         = 5.0     # 5 basis points
MAX_POSITIONS        = 5
MAX_POSITION_PCT     = 0.20    # max 20% of capital per name
MAX_SECTOR_EXPOSURE  = 0.50    # max 50% of capital per sector
MAX_HOLDING_DAYS     = 30      # time-stop

# ---------------------------------------------------------------------------
# RL agent (paper Section 3.4)
# ---------------------------------------------------------------------------
EPSILON_TRAIN = 0.7
EPSILON_TEST  = 0.1

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CACHE_DIR  = './data_cache'
OUTPUT_DIR      = './outputs'
