"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic RNG seeded at 42."""
    return np.random.default_rng(42)


@pytest.fixture(autouse=True)
def _suppress_runtime_warnings() -> None:
    """Some criteria emit harmless RuntimeWarnings on degenerate edge cases."""
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
