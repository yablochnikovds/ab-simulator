"""Unit tests for synthetic data generators."""

from __future__ import annotations

import numpy as np
import pytest

from absim.generators import BinaryGenerator, ContinuousGenerator, RatioGenerator


def test_continuous_marginals():
    gen = ContinuousGenerator(n_per_group=5000, mean=0.0, sd=1.0, rho=0.5)
    rng = np.random.default_rng(0)
    s = gen.sample(rng, mean_shift=0.2)
    assert s.treatment.size == 5000
    assert s.control.size == 5000
    assert abs(s.treatment.mean() - 0.2) < 0.05
    assert abs(s.control.mean()) < 0.05
    assert abs(s.treatment.std() - 1.0) < 0.1


def test_continuous_emits_aux():
    gen = ContinuousGenerator(n_per_group=100, n_strata=4)
    s = gen.sample(np.random.default_rng(0), 0.0)
    assert "covariate_treatment" in s.aux
    assert "strata_treatment" in s.aux
    assert "features_treatment" in s.aux
    assert s.aux["features_treatment"].shape == (100, 1)


def test_continuous_paired_mode_correlates_arms():
    """Paired mode introduces within-pair correlation between T and C."""
    gen = ContinuousGenerator(n_per_group=2000, rho=0.8, paired=True)
    s = gen.sample(np.random.default_rng(0), 0.0)
    corr = float(np.corrcoef(s.treatment, s.control)[0, 1])
    assert corr > 0.4


def test_continuous_unpaired_arms_independent():
    gen = ContinuousGenerator(n_per_group=2000, rho=0.8, paired=False)
    s = gen.sample(np.random.default_rng(0), 0.0)
    corr = float(np.corrcoef(s.treatment, s.control)[0, 1])
    assert abs(corr) < 0.1


@pytest.mark.parametrize("dist", ["normal", "lognormal", "mixture"])
def test_continuous_distributions(dist):
    gen = ContinuousGenerator(n_per_group=1000, distribution=dist)
    s = gen.sample(np.random.default_rng(0), 0.0)
    assert s.treatment.size == 1000
    assert np.isfinite(s.treatment).all()


def test_binary_marginals():
    gen = BinaryGenerator(n_per_group=10000, p=0.1, rho=0.0)
    s = gen.sample(np.random.default_rng(0), mean_shift=0.0)
    assert abs(s.treatment.mean() - 0.1) < 0.02
    assert abs(s.control.mean() - 0.1) < 0.02
    assert set(np.unique(s.treatment)).issubset({0.0, 1.0})


def test_binary_with_shift():
    gen = BinaryGenerator(n_per_group=20000, p=0.1, rho=0.0)
    s = gen.sample(np.random.default_rng(0), mean_shift=0.05)
    assert s.treatment.mean() > s.control.mean()
    assert abs(s.treatment.mean() - 0.15) < 0.02


def test_ratio_generator_emits_pairs():
    gen = RatioGenerator(n_per_group=500, base_rate=0.2, sessions_mean=5.0)
    s = gen.sample(np.random.default_rng(0), mean_shift=0.0)
    assert "numerator_treatment" in s.aux
    assert "denominator_treatment" in s.aux
    n = s.aux["numerator_treatment"]
    d = s.aux["denominator_treatment"]
    assert n.size == 500
    assert (d >= 1).all()


def test_ratio_relative_shift_increases_treatment_ratio():
    gen = RatioGenerator(n_per_group=10000, base_rate=0.2, sessions_mean=5.0, relative=True)
    s = gen.sample(np.random.default_rng(0), mean_shift=0.10)
    ratio_t = s.aux["numerator_treatment"].sum() / s.aux["denominator_treatment"].sum()
    ratio_c = s.aux["numerator_control"].sum() / s.aux["denominator_control"].sum()
    assert ratio_t > ratio_c
    assert abs(ratio_t / ratio_c - 1.10) < 0.05


def test_generators_are_reproducible():
    gen = ContinuousGenerator(n_per_group=200)
    a = gen.sample(np.random.default_rng(123), 0.0)
    b = gen.sample(np.random.default_rng(123), 0.0)
    np.testing.assert_array_equal(a.treatment, b.treatment)


@pytest.mark.parametrize("rho", [0.1, 0.3, 0.5])
def test_binary_generator_realised_correlation_matches_request(rho):
    """Numerical calibration must hit ``rho`` within ±0.02 for moderate values."""
    gen = BinaryGenerator(n_per_group=50_000, p=0.1, rho=rho)
    rng = np.random.default_rng(0)
    s = gen.sample(rng, 0.0)
    realised = float(np.corrcoef(s.control, s.aux["covariate_control"])[0, 1])
    assert abs(realised - rho) < 0.02, (
        f"requested rho={rho}, realised={realised:.4f}; calibration regressed"
    )


@pytest.mark.parametrize(
    ("p", "rho"),
    [(0.1, 0.3), (0.3, 0.3), (0.5, 0.3), (0.5, 0.5), (0.8, 0.3)],
)
def test_binary_calibration_across_baselines(p, rho):
    """Brent-method calibration must hit ``rho`` to ±0.025 across realistic ``p``."""
    gen = BinaryGenerator(n_per_group=50_000, p=p, rho=rho)
    rng = np.random.default_rng(0)
    s = gen.sample(rng, 0.0)
    realised = float(np.corrcoef(s.control, s.aux["covariate_control"])[0, 1])
    assert abs(realised - rho) < 0.025, f"requested rho={rho} at p={p}, realised={realised:.4f}"


def test_binary_saturates_when_request_unreachable():
    """At p=0.05 the logistic link can't reach corr=0.95 — saturate honestly."""
    gen = BinaryGenerator(n_per_group=50_000, p=0.05, rho=0.95)
    rng = np.random.default_rng(0)
    s = gen.sample(rng, 0.0)
    realised = float(np.corrcoef(s.control, s.aux["covariate_control"])[0, 1])
    # Saturate below the request, but stay strictly positive.
    assert 0.0 < realised < 0.95
