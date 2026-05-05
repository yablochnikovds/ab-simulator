"""Unit tests for stratification-based criteria."""

from __future__ import annotations

import numpy as np
import pytest

from absim.criteria import PairedStratification, PostStratification


def test_post_stratification_single_stratum_matches_welch_se_order():
    """With one stratum, post-strat SE should be similar to a t-test SE."""
    rng = np.random.default_rng(0)
    n = 200
    y_t = rng.normal(0.1, 1.0, size=n)
    y_c = rng.normal(0.0, 1.0, size=n)
    strata = np.zeros(n, dtype=int)
    res = PostStratification().test(y_t, y_c, strata_treatment=strata, strata_control=strata)
    # SE ≈ sqrt(2 * 1.0^2 / n) ≈ 0.10
    assert 0.05 < res.std_error < 0.20
    assert res.metadata["n_strata"] == 1


def test_post_stratification_requires_strata_kwargs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="strata"):
        PostStratification().test(rng.normal(size=10), rng.normal(size=10))


def test_post_stratification_size_mismatch():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="match outcome"):
        PostStratification().test(
            rng.normal(size=10),
            rng.normal(size=10),
            strata_treatment=np.zeros(5, dtype=int),
            strata_control=np.zeros(10, dtype=int),
        )


def test_paired_stratification_paired_data():
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(size=n)
    y_t = base + 0.1 + 0.2 * rng.normal(size=n)
    y_c = base + 0.2 * rng.normal(size=n)
    res = PairedStratification(paired=True).test(y_t, y_c)
    assert res.effect == pytest.approx((y_t - y_c).mean())
    assert res.metadata["n_pairs"] == n


def test_paired_stratification_rank_match_fallback():
    rng = np.random.default_rng(1)
    n = 100
    cov_t = rng.normal(size=n)
    cov_c = rng.normal(size=n)
    y_t = cov_t + rng.normal(size=n) * 0.5
    y_c = cov_c + rng.normal(size=n) * 0.5
    res = PairedStratification(paired=False).test(
        y_t, y_c, covariate_treatment=cov_t, covariate_control=cov_c
    )
    assert res.metadata["n_pairs"] == n


def test_paired_stratification_requires_covariate_when_unpaired():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="covariate"):
        PairedStratification(paired=False).test(rng.normal(size=10), rng.normal(size=10))


def test_paired_stratification_requires_min_2():
    with pytest.raises(ValueError, match="at least 2"):
        PairedStratification(paired=True).test(np.array([1.0]), np.array([0.5]))
