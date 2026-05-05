"""Unit tests for ``ZTestProportions``."""

from __future__ import annotations

import numpy as np
import pytest

from absim.criteria import ZTestProportions


def test_textbook_known_z():
    """Two arms with 50 successes / 100 vs 30 successes / 100."""
    t = np.array([1] * 50 + [0] * 50, dtype=float)
    c = np.array([1] * 30 + [0] * 70, dtype=float)
    res = ZTestProportions().test(t, c)
    # Hand-computed: p_pool=0.4, se_pooled = sqrt(0.4*0.6*(1/100+1/100)) ≈ 0.0693.
    # z = 0.2 / 0.0693 ≈ 2.887.
    assert res.statistic == pytest.approx(2.887, abs=0.01)
    assert res.p_value < 0.01
    assert res.rejected


def test_no_difference_does_not_reject():
    rng = np.random.default_rng(0)
    t = (rng.uniform(size=500) < 0.1).astype(float)
    c = (rng.uniform(size=500) < 0.1).astype(float)
    res = ZTestProportions().test(t, c)
    assert not res.rejected
    assert 0.0 <= res.p_value <= 1.0


def test_zero_variance_returns_neutral():
    t = np.zeros(50)
    c = np.zeros(50)
    res = ZTestProportions().test(t, c)
    assert res.p_value == 1.0
    assert res.std_error == 0.0
    assert not res.rejected


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least 1"):
        ZTestProportions().test(np.array([]), np.array([0.0]))


def test_ci_brackets_effect():
    rng = np.random.default_rng(1)
    t = (rng.uniform(size=2000) < 0.12).astype(float)
    c = (rng.uniform(size=2000) < 0.10).astype(float)
    res = ZTestProportions().test(t, c)
    assert res.ci_low <= res.effect <= res.ci_high
