"""Tests for the low-level helpers in ``absim._stats``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from absim._stats import normal_ci, split_seed, t_ci, welch_ttest, wilson_ci


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    t = rng.normal(0.2, 1.0, size=200)
    c = rng.normal(0.0, 1.5, size=180)
    stat, p, eff, se, df = welch_ttest(t, c)
    sp = sp_stats.ttest_ind(t, c, equal_var=False)
    assert eff == pytest.approx(t.mean() - c.mean())
    assert stat == pytest.approx(sp.statistic, rel=1e-10)
    assert p == pytest.approx(sp.pvalue, rel=1e-10)
    assert se > 0
    assert df > 0


def test_welch_handles_zero_variance():
    t = np.ones(10)
    c = np.ones(10)
    stat, p, eff, se, df = welch_ttest(t, c)
    assert se == 0.0
    assert p == 1.0
    assert eff == 0.0
    assert stat == 0.0
    assert df > 0


def test_welch_requires_n_at_least_2():
    with pytest.raises(ValueError, match="at least 2 observations"):
        welch_ttest(np.array([1.0]), np.array([0.0, 1.0]))


def test_normal_and_t_ci_widen_with_alpha():
    """Smaller alpha → wider interval."""
    lo_05, hi_05 = normal_ci(0.0, 1.0, 0.05)
    lo_01, hi_01 = normal_ci(0.0, 1.0, 0.01)
    assert hi_01 - lo_01 > hi_05 - lo_05
    lo_t05, hi_t05 = t_ci(0.0, 1.0, 30, 0.05)
    lo_t01, hi_t01 = t_ci(0.0, 1.0, 30, 0.01)
    assert hi_t01 - lo_t01 > hi_t05 - lo_t05


def test_t_ci_wider_than_normal_at_low_df():
    """Student-t at df=5 must be wider than the normal."""
    lo_n, hi_n = normal_ci(0.0, 1.0, 0.05)
    lo_t, hi_t = t_ci(0.0, 1.0, 5, 0.05)
    assert hi_t - lo_t > hi_n - lo_n


def test_wilson_ci_zero_n():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_extreme_proportions():
    """At boundary p_hat = 1, Wilson must stay inside [0, 1]."""
    lo, hi = wilson_ci(100, 100)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert lo > 0.95


def test_split_seed_independent_streams():
    seeds = split_seed(123, 4)
    rngs = [np.random.default_rng(s) for s in seeds]
    samples = [rng.standard_normal(100) for rng in rngs]
    # Pairwise distinctness.
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            assert not np.allclose(samples[i], samples[j])


def test_split_seed_reproducible():
    a = [np.random.default_rng(s).standard_normal(10) for s in split_seed(7, 3)]
    b = [np.random.default_rng(s).standard_normal(10) for s in split_seed(7, 3)]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_split_seed_accepts_seedsequence():
    ss = np.random.SeedSequence(99)
    seeds = split_seed(ss, 2)
    assert len(seeds) == 2
