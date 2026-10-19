"""Checks for src/eval/stats.py: IQM, bootstrap intervals, the paired test, Holm's correction."""
import numpy as np
import pytest

from src.eval.stats import bootstrap_ci, holm, iqm, paired_bootstrap


def test_iqm_is_scipys_trimmed_mean():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for n in (4, 7, 10, 20):
        x = rng.normal(size=n)
        assert iqm(x) == pytest.approx(scipy_stats.trim_mean(x, 0.25))


def test_iqm_hand_worked():
    # 10 runs: drop the 2 lowest and the 2 highest, average the middle 6.
    x = [0.0, 0.1, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.9, 1.0]
    assert iqm(x) == pytest.approx(np.mean([0.30, 0.31, 0.32, 0.33, 0.34, 0.35]))
    assert iqm(np.array([x, x]), axis=1).shape == (2,)


def test_bootstrap_interval_contains_the_estimate_and_shrinks_with_more_runs():
    rng = np.random.default_rng(1)
    small, large = rng.normal(0.4, 0.02, 10), rng.normal(0.4, 0.02, 100)
    est, lo, hi = bootstrap_ci(small, reps=5000)
    assert lo <= est <= hi
    _, lo2, hi2 = bootstrap_ci(large, reps=5000)
    assert hi2 - lo2 < hi - lo


def test_paired_test_sees_a_real_gain_and_uses_the_pairing():
    rng = np.random.default_rng(2)
    seed_effect = rng.normal(0, 0.05, 10)                  # large differences between seeds, shared by both
    base = 0.35 + seed_effect + rng.normal(0, 0.005, 10)
    better = base + 0.02 + rng.normal(0, 0.005, 10)
    res = paired_bootstrap(better, base, reps=5000)
    assert res["p_value"] < 0.01 and res["low"] > 0 and res["diff"] == pytest.approx(iqm(better) - iqm(base))
    shuffled = paired_bootstrap(better, rng.permutation(base), reps=5000)
    assert shuffled["high"] - shuffled["low"] > 3 * (res["high"] - res["low"]), \
        "with the seeds paired, the shared seed effect cancels and the interval is much narrower"
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2, 3], [1, 2])


def test_holm():
    # m = 4: compare the sorted p-values with 0.05/4, 0.05/3, 0.05/2, 0.05/1 and stop at the first failure.
    p = {"a": 0.010, "b": 0.012, "c": 0.04, "d": 0.001}
    assert holm(p) == {"d": True, "a": True, "b": True, "c": True}
    p = {"a": 0.010, "b": 0.03, "c": 0.04, "d": 0.001}     # b: 0.03 > 0.05/2, so b and c fail
    assert holm(p) == {"d": True, "a": True, "b": False, "c": False}
    p = {"a": 0.02, "b": 0.001}                             # a alone would pass at 0.05, but 0.02 <= 0.05/1 after b
    assert holm(p) == {"b": True, "a": True}
    assert holm({"a": 0.03, "b": 0.03}) == {"a": False, "b": False}, "0.03 > 0.05/2 for the first"
