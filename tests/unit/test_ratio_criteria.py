"""Unit tests for ratio-metric criteria (delta-method and linearization)."""

from __future__ import annotations

import numpy as np
import pytest

from absim.criteria import DeltaMethod, Linearization


def test_delta_method_textbook_constant_denominator():
    """When D is constant, ratio variance reduces to var(N) / D^2 / n."""
    rng = np.random.default_rng(0)
    n = 500
    den_t = np.full(n, 5.0)
    den_c = np.full(n, 5.0)
    num_t = rng.normal(2.0, 1.0, size=n)
    num_c = rng.normal(2.0, 1.0, size=n)
    res = DeltaMethod().test(
        num_t / den_t,
        num_c / den_c,
        numerator_treatment=num_t,
        denominator_treatment=den_t,
        numerator_control=num_c,
        denominator_control=den_c,
    )
    expected_se = float(np.sqrt(2.0 * (1.0 / 25.0) / n))
    assert res.std_error == pytest.approx(expected_se, rel=0.1)


def test_linearization_close_to_delta_method():
    """Linearization and delta-method should give very similar p-values."""
    rng = np.random.default_rng(1)
    n = 1000
    den_t = np.maximum(1, rng.poisson(5.0, size=n))
    den_c = np.maximum(1, rng.poisson(5.0, size=n))
    num_t = rng.poisson(0.2 * den_t)
    num_c = rng.poisson(0.2 * den_c)
    kwargs = {
        "numerator_treatment": num_t,
        "denominator_treatment": den_t,
        "numerator_control": num_c,
        "denominator_control": den_c,
    }
    delta_res = DeltaMethod().test(num_t / den_t, num_c / den_c, **kwargs)
    lin_res = Linearization().test(num_t / den_t, num_c / den_c, **kwargs)
    assert lin_res.p_value == pytest.approx(delta_res.p_value, rel=0.1)


def test_delta_requires_denominator_kwargs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="numerator|denominator"):
        DeltaMethod().test(rng.normal(size=10), rng.normal(size=10))


def test_linearization_requires_kwargs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="numerator|denominator"):
        Linearization().test(rng.normal(size=10), rng.normal(size=10))


def test_delta_method_zero_denominator_raises():
    n = 10
    num = np.zeros(n)
    den = np.zeros(n)
    with pytest.raises(ValueError, match="denominator mean is zero"):
        DeltaMethod().test(
            num,
            num,
            numerator_treatment=num,
            denominator_treatment=den,
            numerator_control=num,
            denominator_control=den,
        )
