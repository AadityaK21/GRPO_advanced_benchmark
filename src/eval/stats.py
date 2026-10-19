"""Aggregate statistics over seeds: IQM, bootstrap intervals, the paired test and Holm's correction.

Follows rliable (Agarwal et al. 2021, arXiv 2108.13264): the interquartile mean (IQM) over runs, with
percentile bootstrap confidence intervals. With one task (GSM8K), rliable's stratified bootstrap is
an ordinary bootstrap over runs. Written in numpy because rliable pins an old `arch` that does not
install on current Python; tests/test_stats.py checks the IQM against scipy's trim_mean.
"""
from __future__ import annotations

import numpy as np


def iqm(x, axis: int = -1) -> np.ndarray:
    """Mean of the middle 50% of values: drop the lowest and highest 25% (rounded down), as
    scipy.stats.trim_mean(x, 0.25) and rliable do. With 10 runs it averages the middle 6."""
    x = np.sort(np.asarray(x, dtype=float), axis=axis)
    n = x.shape[axis]
    cut = int(0.25 * n)
    return np.take(x, np.arange(cut, n - cut), axis=axis).mean(axis=axis)


def bootstrap_ci(x, stat=iqm, reps: int = 50_000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """(statistic, low, high): percentile bootstrap interval over runs (rliable's default, 50k reps)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    boots = stat(x[idx], axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(stat(x)), float(lo), float(hi)


def paired_bootstrap(a, b, stat=iqm, reps: int = 50_000, level: float = 0.95, seed: int = 0) -> dict:
    """Is stat(a) - stat(b) different from 0? a[i] and b[i] are the same seed under two methods.

    Resamples seeds with replacement, the same seeds for both methods, so whatever a seed shares
    across methods (data order, initialisation) cancels. Returns the difference, its percentile
    interval, and a two-sided p-value: twice the smaller share of resampled differences on either
    side of 0 (capped at 1).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired_bootstrap needs two 1-D arrays of equal length, aligned by seed")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    diffs = stat(a[idx], axis=1) - stat(b[idx], axis=1)
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"diff": float(stat(a) - stat(b)), "low": float(lo), "high": float(hi), "p_value": float(p)}


def unpaired_bootstrap(a, b, stat=iqm, reps: int = 10_000, level: float = 0.95, seed: int = 0) -> dict:
    """stat(a) - stat(b) with a percentile interval, resampling each group separately (no pairing)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    da = stat(a[rng.integers(0, len(a), size=(reps, len(a)))], axis=1)
    db = stat(b[rng.integers(0, len(b), size=(reps, len(b)))], axis=1)
    lo, hi = np.quantile(da - db, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"diff": float(stat(a) - stat(b)), "low": float(lo), "high": float(hi)}


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm's step-down correction: which hypotheses are rejected at family-wise level alpha.

    Sort the p-values; the k-th smallest (k from 0) is compared with alpha / (m - k); stop at the
    first that fails, and everything after it is not rejected either.
    """
    order = sorted(p_values, key=p_values.get)
    m, rejected, still = len(order), {}, True
    for k, name in enumerate(order):
        still = still and p_values[name] <= alpha / (m - k)
        rejected[name] = still
    return rejected


def null_control(scores, group: int | None = None, splits: int = 1000, reps: int = 2000, seed: int = 0,
                 stat=iqm, level: float = 0.95) -> float:
    """Share of random splits of `scores` into two groups whose interval for the difference excludes 0.

    scores: the pooled runs of vanilla and its null twin (20 runs: 10 + 10). If the statistics are
    right, this is about 1 - level = 5%. Far more means the intervals are too narrow.
    """
    scores = np.asarray(scores, dtype=float)
    group = group or len(scores) // 2
    rng = np.random.default_rng(seed)
    excluded = 0
    for _ in range(splits):
        perm = rng.permutation(len(scores))
        a, b = scores[perm[:group]], scores[perm[group:2 * group]]
        res = unpaired_bootstrap(a, b, stat=stat, reps=reps, level=level, seed=int(rng.integers(2**31)))
        excluded += res["low"] > 0 or res["high"] < 0
    return excluded / splits
