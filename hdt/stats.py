"""Sharpe-ratio inference under non-normality and multiple testing.

Implements the standard López de Prado / Bailey toolkit:

- Probabilistic Sharpe Ratio (PSR)             -- Bailey and López de Prado (2012)
- Minimum Track Record Length (MinTRL)         -- Bailey and López de Prado (2012)
- Deflated Sharpe Ratio (DSR)                  -- Bailey and López de Prado (2014)
- Combinatorially-Symmetric Cross-Validation,
  Probability of Backtest Overfitting (PBO)    -- Bailey, Borwein, López de Prado, Zhu (2014, 2017)

Formulae use Pearson (non-excess) kurtosis throughout: a Gaussian sample has
kurt=3, not kurt=0. ``scipy.stats.kurtosis`` returns excess kurtosis by
default, so we add 3 in the helper below.

References
----------
Bailey, D.H., López de Prado, M., 2012. The Sharpe ratio efficient frontier.
  Journal of Risk 15 (2), 3-44.
Bailey, D.H., López de Prado, M., 2014. The deflated Sharpe ratio: Correcting
  for selection bias, backtest overfitting and non-normality. Journal of
  Portfolio Management 40 (5), 94-107.
Bailey, D.H., Borwein, J.M., López de Prado, M., Zhu, Q.J., 2017. The
  probability of backtest overfitting. Journal of Computational Finance 20
  (4), 39-69.
López de Prado, M., Lipton, A., Zoonekynd, V., 2025. How to use the Sharpe
  ratio. SSRN 5520741.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _moments(returns: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std_sample, skew, kurt_pearson) for the return series."""
    r = np.asarray(returns, dtype=float)
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    skew = float(sp_stats.skew(r, bias=False)) if r.size > 2 else 0.0
    kurt_pearson = float(sp_stats.kurtosis(r, fisher=False, bias=False)) if r.size > 3 else 3.0
    return mean, std, skew, kurt_pearson


def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = 4,
) -> float:
    """Sample Sharpe ratio in returns' native frequency, then annualised by
    ``sqrt(periods_per_year)``.

    With ``risk_free=0`` this is the "excess-over-zero" convention used in
    the original manuscript. Passing a positive ``risk_free`` (in the same
    frequency as ``returns``) recovers the classical Sharpe ratio.
    """
    r = np.asarray(returns, dtype=float)
    excess = r - risk_free
    s = excess.std(ddof=1)
    return float(excess.mean() / s * math.sqrt(periods_per_year)) if s > 0 else 0.0


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 4,
) -> float:
    """Probability that the true (population) Sharpe ratio exceeds
    ``sr_benchmark``, given the observed sample.

    Both ``sr_benchmark`` and the observed Sharpe are interpreted on the
    *annualised* scale set by ``periods_per_year``.

    Reference: Bailey & López de Prado (2012), eq. (4).
    """
    r = np.asarray(returns, dtype=float)
    T = r.size
    if T < 4:
        return float("nan")
    _, _, skew, kurt = _moments(r)
    sr_hat_per_period = r.mean() / r.std(ddof=1)
    sr_per_period_bench = sr_benchmark / math.sqrt(periods_per_year)
    denom = math.sqrt(max(
        1.0 - skew * sr_hat_per_period + (kurt - 1.0) / 4.0 * sr_hat_per_period ** 2,
        1e-12,
    ))
    z = (sr_hat_per_period - sr_per_period_bench) * math.sqrt(T - 1) / denom
    return float(sp_stats.norm.cdf(z))


def minimum_track_record_length(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    confidence: float = 0.95,
    periods_per_year: int = 4,
) -> float:
    """Minimum number of periods (same frequency as ``returns``) needed for
    PSR(``sr_benchmark``) to reach ``confidence``.

    Returns ``+inf`` when the observed Sharpe does not exceed the benchmark
    -- no finite track record can produce significance in that direction.

    Reference: Bailey & López de Prado (2012), eq. (8).
    """
    r = np.asarray(returns, dtype=float)
    _, _, skew, kurt = _moments(r)
    sr_hat = r.mean() / r.std(ddof=1)
    sr_bench = sr_benchmark / math.sqrt(periods_per_year)
    if sr_hat <= sr_bench:
        return float("inf")
    z_alpha = sp_stats.norm.ppf(confidence)
    variance_correction = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    return 1.0 + variance_correction * (z_alpha / (sr_hat - sr_bench)) ** 2


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe_under_null(n_trials: int, sr_variance: float) -> float:
    """Expected maximum sample Sharpe over ``n_trials`` independent trials
    under H0 of zero true Sharpe.

    Bailey & López de Prado (2014), eq. (8):

        E[max SR_n | H0] ~= sqrt(V[SR]) * { (1 - gamma) * Z[1 - 1/N]
                                            + gamma * Z[1 - 1/(N*e)] }
    """
    if n_trials < 2:
        return 0.0
    z1 = sp_stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sp_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    expected_max_standard = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    return math.sqrt(max(sr_variance, 0.0)) * expected_max_standard


def sharpe_variance(returns: np.ndarray) -> float:
    """Per-period variance of the sample Sharpe estimator, with skew/kurt
    correction (Mertens 2002 / Bailey-LdP).
    """
    r = np.asarray(returns, dtype=float)
    T = r.size
    if T < 4:
        return float("nan")
    _, _, skew, kurt = _moments(r)
    sr_hat = r.mean() / r.std(ddof=1)
    return (1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2) / (T - 1)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 4,
) -> float:
    """Deflated Sharpe ratio: PSR evaluated against the expected best-of-N
    Sharpe under H0, instead of against zero.

    A DSR > 0.95 indicates that the observed Sharpe is unlikely (at the 5%
    level) to be the chance maximum of ``n_trials`` independent backtests.

    Reference: Bailey & López de Prado (2014), eqs. (8)-(9).
    """
    if n_trials < 2:
        return probabilistic_sharpe_ratio(returns, 0.0, periods_per_year)
    sr_var = sharpe_variance(returns)
    sr0_per_period = expected_max_sharpe_under_null(n_trials, sr_var)
    sr0_annualised = sr0_per_period * math.sqrt(periods_per_year)
    return probabilistic_sharpe_ratio(returns, sr0_annualised, periods_per_year)


# ---------------------------------------------------------------------------
# CSCV / Probability of Backtest Overfitting
# ---------------------------------------------------------------------------

@dataclass
class CSCVResult:
    """Result of a Combinatorially Symmetric Cross-Validation run."""

    pbo: float
    n_splits: int
    in_sample_sharpes: List[float] = field(default_factory=list)
    out_sample_sharpes: List[float] = field(default_factory=list)
    logits: List[float] = field(default_factory=list)


def combinatorially_symmetric_cv(
    performance_matrix: np.ndarray,
    n_slices: int = 16,
) -> CSCVResult:
    """Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    performance_matrix : (T, N) array
        Per-period returns for each of N candidate configurations across T
        periods. The procedure requires N >= 2.
    n_slices : int
        Number of equal-length slices to partition the T periods into.
        Must be even (so half can be assigned to IS and half to OOS).
        Bailey-Borwein-LdP-Zhu (2017) recommend S = 16.

    Returns
    -------
    CSCVResult with attribute ``pbo`` -- the proportion of splits in which
    the best in-sample configuration finishes below the median in the
    complementary out-of-sample slice.
    """
    M = np.asarray(performance_matrix, dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError("CSCV requires at least 2 candidate configurations.")
    if n_slices % 2:
        raise ValueError("n_slices must be even.")

    # Partition rows into n_slices equal-length groups (trim trailing rows).
    slice_size = T // n_slices
    if slice_size == 0:
        raise ValueError("T must be at least n_slices.")
    trimmed = M[: slice_size * n_slices]
    slices = trimmed.reshape(n_slices, slice_size, N)

    indices = list(range(n_slices))
    half = n_slices // 2
    is_sharpes: List[float] = []
    oos_sharpes: List[float] = []
    logits: List[float] = []
    overfit_count = 0
    total = 0

    for is_index in combinations(indices, half):
        is_set = set(is_index)
        oos_set = [i for i in indices if i not in is_set]
        is_data = slices[list(is_index)].reshape(-1, N)
        oos_data = slices[oos_set].reshape(-1, N)

        is_sr = is_data.mean(axis=0) / (is_data.std(axis=0, ddof=1) + 1e-12)
        oos_sr = oos_data.mean(axis=0) / (oos_data.std(axis=0, ddof=1) + 1e-12)

        winner = int(np.argmax(is_sr))
        winner_oos = oos_sr[winner]
        oos_rank = sp_stats.rankdata(oos_sr)[winner]  # 1..N
        relative_rank = oos_rank / (N + 1)
        # Logit of OOS rank; below 0 means below-median OOS
        logits.append(float(math.log(relative_rank / (1.0 - relative_rank))))
        is_sharpes.append(float(is_sr[winner]))
        oos_sharpes.append(float(winner_oos))

        if oos_rank <= N / 2:
            overfit_count += 1
        total += 1

    pbo = overfit_count / total if total else float("nan")
    return CSCVResult(
        pbo=pbo,
        n_splits=total,
        in_sample_sharpes=is_sharpes,
        out_sample_sharpes=oos_sharpes,
        logits=logits,
    )


def pbo_from_train_test_pairs(train_returns: np.ndarray, test_returns: np.ndarray) -> float:
    """Lightweight overfitting diagnostic when only the realised single
    strategy's per-fold (train, test) returns are available.

    Reports the fraction of folds in which a top-half training return is
    followed by a bottom-half test return. This is *not* full CSCV (which
    requires a family of N strategies), but gives a comparable summary
    when N = 1.
    """
    tr = np.asarray(train_returns, dtype=float)
    te = np.asarray(test_returns, dtype=float)
    if tr.shape != te.shape or tr.size == 0:
        return float("nan")
    tr_above = tr >= np.median(tr)
    te_below = te < np.median(te)
    if not tr_above.any():
        return float("nan")
    return float((tr_above & te_below).sum() / tr_above.sum())


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def sharpe_inference_table(
    returns: np.ndarray,
    n_trials_options: Sequence[int] = (1, 10, 30, 100),
    risk_free_per_period: float = 0.0,
    periods_per_year: int = 4,
    train_returns: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build a one-table summary suitable for inclusion in the manuscript."""
    r = np.asarray(returns, dtype=float)
    sr_zero = sharpe_ratio(r, risk_free=0.0, periods_per_year=periods_per_year)
    sr_rf = sharpe_ratio(r, risk_free=risk_free_per_period, periods_per_year=periods_per_year)
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, periods_per_year=periods_per_year)
    mintrl = minimum_track_record_length(r, periods_per_year=periods_per_year)
    rows = [
        ("Sharpe ratio (excess over zero, ann.)", sr_zero),
        ("Sharpe ratio (excess over Rf, ann.)",   sr_rf),
        ("Skewness",                              _moments(r)[2]),
        ("Kurtosis (Pearson)",                    _moments(r)[3]),
        ("PSR (SR* = 0)",                         psr),
        ("MinTRL @ 95% (periods)",                mintrl),
    ]
    for n in n_trials_options:
        rows.append((f"DSR (N = {n} trials)", deflated_sharpe_ratio(r, n_trials=n,
                                                                    periods_per_year=periods_per_year)))
    if train_returns is not None:
        rows.append(("PBO (single-strategy train/test rank flip)",
                     pbo_from_train_test_pairs(train_returns, returns)))
    return pd.DataFrame(rows, columns=["Statistic", "Value"])
