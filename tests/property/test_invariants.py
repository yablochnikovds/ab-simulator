"""Property-based tests for criterion invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from absim.criteria import (
    CUPED,
    Bootstrap,
    DeltaMethod,
    Linearization,
    PairedStratification,
    WelchTTest,
)

# Vectors of moderate size with finite values.
_finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_vec = arrays(np.float64, shape=st.integers(min_value=10, max_value=200), elements=_finite)
_vec_large = arrays(np.float64, shape=st.integers(min_value=20, max_value=200), elements=_finite)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(t=_vec, c=_vec)
def test_welch_invariant_to_unit_permutation(t, c):
    """Permuting unit indices within each arm must not change the result."""
    if len(t) < 2 or len(c) < 2 or np.std(t) == 0 or np.std(c) == 0:
        return
    rng = np.random.default_rng(0)
    res_a = WelchTTest().test(t, c)
    res_b = WelchTTest().test(rng.permutation(t), rng.permutation(c))
    assert res_a.statistic == pytest.approx(res_b.statistic, rel=1e-9, abs=1e-12)
    assert res_a.p_value == pytest.approx(res_b.p_value, rel=1e-9, abs=1e-12)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(t=_vec, c=_vec)
def test_welch_p_value_in_unit_interval(t, c):
    if len(t) < 2 or len(c) < 2:
        return
    res = WelchTTest().test(t, c)
    assert 0.0 <= res.p_value <= 1.0


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(t=_vec_large, c=_vec_large)
def test_cuped_with_zero_correlation_close_to_welch(t, c):
    """With covariate orthogonal to outcome, theta ≈ 0 → CUPED ≈ Welch."""
    if len(t) < 5 or len(c) < 5 or np.std(t) == 0 or np.std(c) == 0:
        return
    rng = np.random.default_rng(0)
    cov_t = rng.standard_normal(t.size)
    cov_c = rng.standard_normal(c.size)
    cuped = CUPED().test(t, c, covariate_treatment=cov_t, covariate_control=cov_c)
    welch = WelchTTest().test(t, c)
    # CUPED's adjustment is small but not exactly zero — they should be close
    # in p-value space but not identical. Use a generous tolerance.
    assert abs(cuped.p_value - welch.p_value) < 0.3


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(t=_vec_large, c=_vec_large)
def test_bootstrap_ci_brackets_observed(t, c):
    if len(t) < 5 or len(c) < 5 or np.std(t) == 0 or np.std(c) == 0:
        return
    res = Bootstrap(n_resamples=200, method="percentile", seed=0).test(t, c)
    # Allow tiny numerical wiggle on equality with the observed effect.
    assert res.ci_low - 1e-9 <= res.effect <= res.ci_high + 1e-9


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scale=st.floats(min_value=0.5, max_value=10.0, allow_nan=False))
def test_delta_method_symmetric_in_scale(scale):
    """Scaling both N and D by the same constant cancels in the ratio."""
    rng = np.random.default_rng(0)
    n = 100
    den_t = rng.uniform(1.0, 5.0, size=n)
    den_c = rng.uniform(1.0, 5.0, size=n)
    num_t = den_t * 0.3 + rng.normal(scale=0.1, size=n)
    num_c = den_c * 0.3 + rng.normal(scale=0.1, size=n)
    base = DeltaMethod().test(
        num_t / den_t,
        num_c / den_c,
        numerator_treatment=num_t,
        denominator_treatment=den_t,
        numerator_control=num_c,
        denominator_control=den_c,
    )
    scaled = DeltaMethod().test(
        num_t / den_t,
        num_c / den_c,
        numerator_treatment=num_t * scale,
        denominator_treatment=den_t * scale,
        numerator_control=num_c * scale,
        denominator_control=den_c * scale,
    )
    # Effect is unchanged (ratio invariant); SE scales 1/scale^0 = invariant too.
    assert base.effect == pytest.approx(scaled.effect, rel=1e-9, abs=1e-12)
    assert base.std_error == pytest.approx(scaled.std_error, rel=1e-9, abs=1e-12)


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(t=_vec, c=_vec)
def test_paired_paired_mode_uses_index_pairing(t, c):
    """In paired=True mode, result depends only on the index-aligned differences."""
    n = min(len(t), len(c))
    if n < 2:
        return
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    a = PairedStratification(paired=True).test(t[:n], c[:n])
    # If we permute T and C with the SAME permutation, differences are reordered
    # but mean and variance are unchanged.
    b = PairedStratification(paired=True).test(t[:n][perm], c[:n][perm])
    assert a.effect == pytest.approx(b.effect, rel=1e-9, abs=1e-12)
    assert a.p_value == pytest.approx(b.p_value, rel=1e-9, abs=1e-12)


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scale=st.floats(min_value=0.5, max_value=5.0, allow_nan=False))
def test_linearization_close_to_delta(scale):
    rng = np.random.default_rng(0)
    n = 200
    den_t = rng.uniform(1.0, 5.0, size=n) * scale
    den_c = rng.uniform(1.0, 5.0, size=n) * scale
    num_t = den_t * 0.3 + rng.normal(scale=0.1, size=n)
    num_c = den_c * 0.3 + rng.normal(scale=0.1, size=n)
    kw = {
        "numerator_treatment": num_t,
        "denominator_treatment": den_t,
        "numerator_control": num_c,
        "denominator_control": den_c,
    }
    delta = DeltaMethod().test(num_t / den_t, num_c / den_c, **kw)
    lin = Linearization().test(num_t / den_t, num_c / den_c, **kw)
    # P-values should agree to ~10% tolerance — same asymptotic distribution.
    assert lin.p_value == pytest.approx(delta.p_value, rel=0.2, abs=0.05)
