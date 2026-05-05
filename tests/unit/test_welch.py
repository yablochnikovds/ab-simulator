"""Unit tests for ``absim.criteria.WelchTTest``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from absim.criteria import WelchTTest


def test_textbook_handcomputed_p_value():
    """Welch on a tiny dataset must agree with scipy's ttest_ind(equal_var=False)."""
    t = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    c = np.array([4.0, 4.5, 5.0, 5.5, 6.0])
    res = WelchTTest().test(t, c)
    sp = sp_stats.ttest_ind(t, c, equal_var=False)
    assert res.p_value == pytest.approx(sp.pvalue, rel=1e-10)
    assert res.statistic == pytest.approx(sp.statistic, rel=1e-10)
    assert res.effect == pytest.approx(t.mean() - c.mean())


def test_rejects_obvious_difference():
    """Two arms with disjoint distributions must give p < alpha."""
    rng = np.random.default_rng(0)
    t = rng.normal(5.0, 0.5, size=100)
    c = rng.normal(0.0, 0.5, size=100)
    res = WelchTTest(alpha=0.05).test(t, c)
    assert res.rejected
    assert res.p_value < 0.001


def test_no_difference_does_not_reject():
    rng = np.random.default_rng(1)
    t = rng.normal(0.0, 1.0, size=50)
    c = rng.normal(0.0, 1.0, size=50)
    res = WelchTTest().test(t, c)
    # Sometimes by chance it can reject — but with seed=1 it doesn't.
    assert not res.rejected


def test_ci_brackets_effect():
    rng = np.random.default_rng(2)
    t = rng.normal(0.5, 1.0, size=200)
    c = rng.normal(0.0, 1.0, size=200)
    res = WelchTTest().test(t, c)
    assert res.ci_low <= res.effect <= res.ci_high


def test_invariant_to_unit_permutation():
    """Permuting unit indices within each arm changes nothing."""
    rng = np.random.default_rng(3)
    t = rng.normal(0.1, 1.0, size=80)
    c = rng.normal(0.0, 1.0, size=80)
    perm_t = rng.permutation(t)
    perm_c = rng.permutation(c)
    a = WelchTTest().test(t, c)
    b = WelchTTest().test(perm_t, perm_c)
    assert a.p_value == pytest.approx(b.p_value)
    assert a.statistic == pytest.approx(b.statistic)


def test_zero_variance_returns_no_signal():
    t = np.ones(10)
    c = np.ones(10)
    res = WelchTTest().test(t, c)
    assert not res.rejected
    assert res.p_value == 1.0
    assert res.std_error == 0.0


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match="at least 2"):
        WelchTTest().test(np.array([1.0]), np.array([2.0, 3.0]))
