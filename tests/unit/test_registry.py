"""Tests for the criterion registry."""

from __future__ import annotations

import pytest

from absim.criteria import available, get


def test_all_expected_criteria_registered():
    expected = {
        "welch_t",
        "z_proportion",
        "cuped",
        "cupac",
        "bootstrap",
        "delta_method",
        "linearization",
        "post_stratification",
        "paired_stratification",
    }
    assert expected.issubset(set(available()))


def test_get_unknown_criterion_raises():
    with pytest.raises(KeyError, match="unknown criterion"):
        get("does_not_exist")


def test_get_returns_class():
    cls = get("welch_t")
    assert callable(cls)


def test_register_duplicate_raises():
    from absim.criteria.base import register

    @register("dummy_unique_for_test")
    class _Dummy:
        name = "dummy_unique_for_test"

        def test(self, t, c, **kw):  # pragma: no cover - shape only
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        register("dummy_unique_for_test")(_Dummy)
