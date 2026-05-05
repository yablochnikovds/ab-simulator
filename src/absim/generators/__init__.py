"""Synthetic-data generators for absim."""

from __future__ import annotations

from absim.generators.base import Generator, Sample
from absim.generators.binary import BinaryGenerator
from absim.generators.continuous import ContinuousGenerator
from absim.generators.ratio import RatioGenerator

__all__ = [
    "BinaryGenerator",
    "ContinuousGenerator",
    "Generator",
    "RatioGenerator",
    "Sample",
]
