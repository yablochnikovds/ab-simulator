"""Unit tests for :class:`absim.generators.EmpiricalGenerator`."""

from __future__ import annotations

import numpy as np
import pytest

from absim.generators import EmpiricalGenerator, Sample

# ----------------------------- validation ----------------------------------


def test_requires_at_least_2_observations():
    with pytest.raises(ValueError, match="at least 2"):
        EmpiricalGenerator(outcomes=np.array([1.0]))


def test_covariate_length_must_match():
    with pytest.raises(ValueError, match="`covariate`"):
        EmpiricalGenerator(outcomes=np.array([1.0, 2.0, 3.0]), covariate=np.array([0.5, 0.6]))


def test_strata_length_must_match():
    with pytest.raises(ValueError, match="`strata`"):
        EmpiricalGenerator(outcomes=np.array([1.0, 2.0, 3.0]), strata=np.array([0, 1]))


def test_ratio_requires_both_arrays():
    with pytest.raises(ValueError, match="`numerator` and `denominator`"):
        EmpiricalGenerator(outcomes=np.array([0.1, 0.2, 0.3]), numerator=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="`numerator` and `denominator`"):
        EmpiricalGenerator(
            outcomes=np.array([0.1, 0.2, 0.3]), denominator=np.array([1.0, 2.0, 3.0])
        )


def test_zero_denominator_sum_rejected():
    with pytest.raises(ValueError, match="denominator sum is zero"):
        EmpiricalGenerator(
            outcomes=np.array([0.0, 0.0, 0.0]),
            numerator=np.array([0.0, 0.0, 0.0]),
            denominator=np.array([0.0, 0.0, 0.0]),
        )


def test_n_per_group_at_least_2():
    with pytest.raises(ValueError, match="n_per_group"):
        EmpiricalGenerator(outcomes=np.array([1.0, 2.0]), n_per_group=1)


# ----------------------------- mode detection ------------------------------


def test_detects_binary_outcome():
    gen = EmpiricalGenerator(outcomes=np.array([0.0, 1.0, 0.0, 1.0, 0.0]))
    assert gen._is_binary
    assert not gen._is_ratio


def test_detects_ratio_mode_when_both_numerator_and_denominator():
    gen = EmpiricalGenerator(
        outcomes=np.array([0.5, 0.6, 0.4]),
        numerator=np.array([5.0, 6.0, 4.0]),
        denominator=np.array([10.0, 10.0, 10.0]),
    )
    assert gen._is_ratio
    assert not gen._is_binary


def test_continuous_mode_when_no_special_structure():
    gen = EmpiricalGenerator(outcomes=np.array([0.5, 1.5, 2.5, 3.5]))
    assert not gen._is_binary
    assert not gen._is_ratio


# ----------------------------- shape & aux contents ------------------------


def test_sample_shape_continuous():
    rng = np.random.default_rng(0)
    gen = EmpiricalGenerator(outcomes=np.linspace(1.0, 5.0, 50), n_per_group=200, name="hist")
    sample = gen.sample(rng, mean_shift=0.0)
    assert isinstance(sample, Sample)
    assert sample.treatment.shape == (200,)
    assert sample.control.shape == (200,)
    assert sample.aux == {}


def test_aux_includes_covariate_and_features():
    rng = np.random.default_rng(0)
    gen = EmpiricalGenerator(
        outcomes=np.arange(20.0),
        covariate=np.arange(20.0) + 100.0,
        n_per_group=50,
    )
    s = gen.sample(rng, mean_shift=0.0)
    assert s.aux["covariate_treatment"].shape == (50,)
    assert s.aux["covariate_control"].shape == (50,)
    assert s.aux["features_treatment"].shape == (50, 1)
    assert s.aux["features_control"].shape == (50, 1)


def test_aux_includes_strata_when_provided():
    rng = np.random.default_rng(0)
    gen = EmpiricalGenerator(outcomes=np.arange(20.0), strata=np.arange(20) % 4, n_per_group=80)
    s = gen.sample(rng, mean_shift=0.0)
    assert "strata_treatment" in s.aux
    assert s.aux["strata_treatment"].dtype.kind in ("i", "u")


def test_strata_derived_from_covariate_when_n_strata_gt_1():
    rng = np.random.default_rng(0)
    gen = EmpiricalGenerator(
        outcomes=np.arange(40.0),
        covariate=np.arange(40.0),
        n_per_group=80,
        n_strata=4,
    )
    s = gen.sample(rng, mean_shift=0.0)
    assert "strata_treatment" in s.aux
    # 4 distinct strata expected with rough balance
    unique_strata = np.unique(s.aux["strata_treatment"])
    assert 1 <= unique_strata.size <= 4


def test_aux_includes_numerator_and_denominator_in_ratio_mode():
    rng = np.random.default_rng(0)
    gen = EmpiricalGenerator(
        outcomes=np.array([0.1, 0.2, 0.3]),
        numerator=np.array([1.0, 2.0, 3.0]),
        denominator=np.array([10.0, 10.0, 10.0]),
        n_per_group=30,
    )
    s = gen.sample(rng, mean_shift=0.0)
    assert s.aux["numerator_treatment"].shape == (30,)
    assert s.aux["denominator_treatment"].shape == (30,)


# ----------------------------- effect injection ----------------------------


def test_continuous_absolute_shift_increases_mean():
    rng = np.random.default_rng(0)
    gen = EmpiricalGenerator(outcomes=np.linspace(0.0, 10.0, 1000), n_per_group=500, relative=False)
    s = gen.sample(rng, mean_shift=2.5)
    # Expected: control mean ~5, treatment mean ~7.5
    assert abs(s.treatment.mean() - 7.5) < 0.5
    assert abs(s.control.mean() - 5.0) < 0.5


def test_continuous_relative_shift_multiplies():
    rng = np.random.default_rng(0)
    gen = EmpiricalGenerator(outcomes=np.full(500, 4.0), n_per_group=500, relative=True)
    s = gen.sample(rng, mean_shift=0.25)
    # relative: 4.0 * 1.25 = 5.0
    assert abs(s.treatment.mean() - 5.0) < 1e-9
    assert abs(s.control.mean() - 4.0) < 1e-9


def test_binary_shift_moves_proportion_in_expected_direction():
    rng = np.random.default_rng(0)
    outcomes = (np.arange(10000) < 1000).astype(float)  # baseline p = 0.1
    gen = EmpiricalGenerator(outcomes=outcomes, n_per_group=20000)
    s = gen.sample(rng, mean_shift=0.05)  # target p ~ 0.15
    # Should be roughly 0.15
    assert 0.13 < s.treatment.mean() < 0.17
    assert 0.09 < s.control.mean() < 0.11


def test_binary_negative_shift_flips_ones_to_zeros():
    rng = np.random.default_rng(0)
    outcomes = (np.arange(10000) < 5000).astype(float)  # baseline p = 0.5
    gen = EmpiricalGenerator(outcomes=outcomes, n_per_group=20000)
    s = gen.sample(rng, mean_shift=-0.2)  # target p ~ 0.3
    assert 0.28 < s.treatment.mean() < 0.32


def test_ratio_relative_lift_shifts_realised_ratio():
    rng = np.random.default_rng(0)
    n = 500
    sessions = np.full(n, 10.0)
    clicks = np.full(n, 2.0)  # baseline ratio 0.2
    gen = EmpiricalGenerator(
        outcomes=clicks / sessions,
        numerator=clicks,
        denominator=sessions,
        n_per_group=2000,
        relative=True,
    )
    s = gen.sample(rng, mean_shift=0.5)  # target ratio = 0.3
    realised_t = s.aux["numerator_treatment"].sum() / s.aux["denominator_treatment"].sum()
    assert abs(realised_t - 0.3) < 1e-9


# ----------------------------- reproducibility -----------------------------


def test_same_seed_produces_same_sample():
    gen = EmpiricalGenerator(
        outcomes=np.arange(100.0), covariate=np.arange(100.0) + 1.0, n_per_group=50
    )
    s1 = gen.sample(np.random.default_rng(42), mean_shift=0.5)
    s2 = gen.sample(np.random.default_rng(42), mean_shift=0.5)
    np.testing.assert_array_equal(s1.treatment, s2.treatment)
    np.testing.assert_array_equal(s1.control, s2.control)
    np.testing.assert_array_equal(s1.aux["covariate_treatment"], s2.aux["covariate_treatment"])


def test_indices_paired_across_outcome_and_covariate():
    """Bootstrap must resample outcome and covariate at the same indices to
    preserve the within-unit (Y, X) correlation."""
    rng = np.random.default_rng(7)
    n = 200
    outcomes = np.arange(n, dtype=float)
    covariate = outcomes * 2.0 + 0.0  # outcome and covariate are perfectly paired
    gen = EmpiricalGenerator(outcomes=outcomes, covariate=covariate, n_per_group=500)
    s = gen.sample(rng, mean_shift=0.0)
    # If indices are paired correctly, covariate_control == 2 * control elementwise
    np.testing.assert_array_equal(s.aux["covariate_control"], s.control * 2.0)
