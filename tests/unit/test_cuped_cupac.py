"""Unit tests for CUPED and CUPAC."""

from __future__ import annotations

import numpy as np
import pytest

from absim.criteria import CUPAC, CUPED, WelchTTest


def test_cuped_with_zero_correlation_matches_welch():
    """When the covariate is independent of Y, theta ≈ 0 → result ≈ Welch."""
    rng = np.random.default_rng(0)
    n = 500
    y_t = rng.normal(0.1, 1.0, size=n)
    y_c = rng.normal(0.0, 1.0, size=n)
    x_t = rng.normal(0.0, 1.0, size=n)
    x_c = rng.normal(0.0, 1.0, size=n)
    cuped = CUPED().test(y_t, y_c, covariate_treatment=x_t, covariate_control=x_c)
    welch = WelchTTest().test(y_t, y_c)
    # Theta is small; CUPED p-value should be very close to Welch's.
    assert cuped.p_value == pytest.approx(welch.p_value, abs=0.05)


def test_cuped_reduces_se_when_covariate_is_informative():
    """A strong covariate must shrink the standard error."""
    rng = np.random.default_rng(0)
    n = 1000
    x_t = rng.normal(size=n)
    x_c = rng.normal(size=n)
    y_t = 0.0 + 0.9 * x_t + 0.4 * rng.normal(size=n)
    y_c = 0.0 + 0.9 * x_c + 0.4 * rng.normal(size=n)
    welch = WelchTTest().test(y_t, y_c)
    cuped = CUPED().test(y_t, y_c, covariate_treatment=x_t, covariate_control=x_c)
    assert cuped.std_error < welch.std_error * 0.7


def test_cuped_requires_covariate_kwargs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="covariate_treatment"):
        CUPED().test(rng.normal(size=10), rng.normal(size=10))


def test_cuped_size_mismatch():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="match outcome array sizes"):
        CUPED().test(
            rng.normal(size=10),
            rng.normal(size=10),
            covariate_treatment=rng.normal(size=5),
            covariate_control=rng.normal(size=10),
        )


def test_cupac_accepts_precomputed_predictions():
    """CUPAC with precomputed preds is just CUPED on those preds."""
    rng = np.random.default_rng(0)
    n = 500
    y_t = rng.normal(0.0, 1.0, size=n)
    y_c = rng.normal(0.0, 1.0, size=n)
    pred_t = rng.normal(0.0, 1.0, size=n)
    pred_c = rng.normal(0.0, 1.0, size=n)
    res = CUPAC().test(y_t, y_c, prediction_treatment=pred_t, prediction_control=pred_c)
    cuped = CUPED().test(y_t, y_c, covariate_treatment=pred_t, covariate_control=pred_c)
    assert res.p_value == pytest.approx(cuped.p_value, rel=1e-9)


def test_cupac_with_features_runs_oof():
    rng = np.random.default_rng(0)
    n = 200
    feat_t = rng.normal(size=(n, 3))
    feat_c = rng.normal(size=(n, 3))
    y_t = feat_t @ np.array([0.5, -0.3, 0.2]) + rng.normal(size=n) * 0.5
    y_c = feat_c @ np.array([0.5, -0.3, 0.2]) + rng.normal(size=n) * 0.5
    res = CUPAC(n_splits=5).test(y_t, y_c, features_treatment=feat_t, features_control=feat_c)
    assert res.std_error >= 0
    assert "model" in res.metadata


def test_cupac_requires_predictions_or_features():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="prediction|features"):
        CUPAC().test(rng.normal(size=10), rng.normal(size=10))


def test_cupac_pooled_oof_unbiased_under_h1():
    """Pooled OOF must preserve the average treatment effect under H₁."""
    rng = np.random.default_rng(0)
    n = 1000
    feat = rng.normal(size=(2 * n, 3))
    feat_t, feat_c = feat[:n], feat[n:]
    base = feat @ np.array([0.5, -0.3, 0.2])
    y_t = base[:n] + 0.5 + rng.normal(size=n) * 0.5  # +0.5 effect
    y_c = base[n:] + rng.normal(size=n) * 0.5
    res = CUPAC(n_splits=5).test(y_t, y_c, features_treatment=feat_t, features_control=feat_c)
    assert abs(res.effect - 0.5) < 0.1, (
        f"pooled-OOF must preserve ATE; got effect={res.effect:.4f} (expected 0.5)"
    )
