"""Unit tests for the bootstrap criterion."""

from __future__ import annotations

import numpy as np
import pytest

from absim.criteria import Bootstrap


def test_percentile_ci_brackets_observed_effect():
    rng = np.random.default_rng(0)
    t = rng.normal(0.5, 1.0, size=200)
    c = rng.normal(0.0, 1.0, size=200)
    boot = Bootstrap(n_resamples=400, method="percentile", seed=0)
    res = boot.test(t, c)
    assert res.ci_low <= res.effect <= res.ci_high


def test_bca_ci_brackets_observed_effect():
    rng = np.random.default_rng(1)
    t = rng.exponential(1.0, size=200)
    c = rng.exponential(1.0, size=200)
    boot = Bootstrap(n_resamples=400, method="bca", seed=1)
    res = boot.test(t, c)
    assert res.ci_low <= res.effect <= res.ci_high


def test_rejects_obvious_difference():
    rng = np.random.default_rng(2)
    t = rng.normal(2.0, 1.0, size=300)
    c = rng.normal(0.0, 1.0, size=300)
    res = Bootstrap(n_resamples=500, method="percentile", seed=2).test(t, c)
    assert res.rejected


def test_does_not_reject_under_null():
    rng = np.random.default_rng(3)
    t = rng.normal(0.0, 1.0, size=200)
    c = rng.normal(0.0, 1.0, size=200)
    res = Bootstrap(n_resamples=500, method="percentile", seed=3).test(t, c)
    assert not res.rejected


def test_method_metadata_propagated():
    rng = np.random.default_rng(0)
    t = rng.normal(size=50)
    c = rng.normal(size=50)
    res = Bootstrap(n_resamples=100, method="bca", seed=0).test(t, c)
    assert res.metadata["method"] == "bca"
    assert res.metadata["n_resamples"] == 100


def test_passing_external_rng_is_deterministic():
    """When the simulator-style rng kwarg is given, results are reproducible."""
    rng = np.random.default_rng(7)
    t = rng.normal(size=100)
    c = rng.normal(size=100)
    boot = Bootstrap(n_resamples=300, method="percentile")
    a = boot.test(t, c, rng=np.random.default_rng(99))
    b = boot.test(t, c, rng=np.random.default_rng(99))
    assert a.ci_low == pytest.approx(b.ci_low)
    assert a.ci_high == pytest.approx(b.ci_high)


# --------------------------------------------------------------------------- #
# Direct tests for the BCa-adjusted p-value helper introduced in 0.2.0.
# --------------------------------------------------------------------------- #


def test_bca_pvalue_near_one_when_zero_at_centre_of_boot_dist():
    """Symmetric boot distribution with mean 0 → p-value ≈ 1."""
    from absim.criteria.bootstrap import _bca_two_sided_pvalue

    rng = np.random.default_rng(0)
    boot = rng.normal(loc=0.0, scale=1.0, size=5000)
    jack = np.zeros(50)  # a_hat = 0 (no acceleration)
    p = _bca_two_sided_pvalue(boot, observed=0.0, jack=jack)
    assert 0.7 <= p <= 1.0


def test_bca_pvalue_small_when_zero_in_far_tail():
    """Boot distribution centred well above 0 → tiny p-value."""
    from absim.criteria.bootstrap import _bca_two_sided_pvalue

    rng = np.random.default_rng(0)
    boot = rng.normal(loc=2.0, scale=0.3, size=5000)
    jack = np.zeros(50)
    p = _bca_two_sided_pvalue(boot, observed=2.0, jack=jack)
    assert p < 1e-3


def test_bca_rejection_matches_pvalue_threshold():
    """For Bootstrap(method='bca') the reported `rejected` must match `p_value < alpha`."""
    rng = np.random.default_rng(0)
    t = rng.normal(0.5, 1.0, size=200)
    c = rng.normal(0.0, 1.0, size=200)
    res = Bootstrap(n_resamples=2000, method="bca", seed=0).test(t, c, rng=rng)
    assert res.rejected == (res.p_value < 0.05)
