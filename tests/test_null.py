"""The null control's machinery: when nothing differs, about 5% of intervals should exclude zero.

The real null control runs on results (vanilla's 10 runs pooled with its null twin's 10). This test
checks the statistics themselves on simulated runs where the truth is known: if it fails, no interval
in the paper can be trusted.
"""
import numpy as np

from src.eval.stats import null_control, paired_bootstrap, unpaired_bootstrap


def test_vanilla_vs_vanilla_intervals_exclude_zero_about_5_percent_of_the_time():
    rng = np.random.default_rng(0)
    pools, excluded = 1500, 0
    for i in range(pools):                     # independent pools: one random split each
        runs = 0.35 + rng.normal(0, 0.02, 20)  # 20 runs, same method, accuracy around 35%
        perm = rng.permutation(20)
        res = unpaired_bootstrap(runs[perm[:10]], runs[perm[10:]], reps=2000, seed=i)
        excluded += res["low"] > 0 or res["high"] < 0
    rate = excluded / pools
    # Measured 4.6% when this test was written. Binomial noise over 1500 pools is about +-1.1%.
    assert 0.025 <= rate <= 0.08, f"{rate:.1%} of null intervals exclude zero; expected about 5%"


def test_the_paired_test_rejects_about_5_percent_under_the_null():
    rng = np.random.default_rng(1)
    trials, rejected = 1500, 0
    for i in range(trials):
        seed_effect = rng.normal(0, 0.02, 10)
        a = 0.35 + seed_effect + rng.normal(0, 0.02, 10)
        b = 0.35 + seed_effect + rng.normal(0, 0.02, 10)
        rejected += paired_bootstrap(a, b, reps=2000, seed=i)["p_value"] < 0.05
    rate = rejected / trials
    # Measured 6.0-6.5% when written: a little above 5%, as expected of a bootstrap at 10 seeds.
    assert 0.025 <= rate <= 0.09, f"{rate:.1%} false rejections; expected about 5%"


def test_broken_statistics_are_caught():
    """null_control must flag intervals that are too narrow. A 50% interval stands in for a bug that
    understates the spread between seeds."""
    rng = np.random.default_rng(2)
    runs = 0.35 + rng.normal(0, 0.02, 20)
    assert null_control(runs, splits=300, reps=1000) <= 0.15
    assert null_control(runs, splits=300, reps=1000, level=0.5) > 0.3
