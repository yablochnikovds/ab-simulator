"""Generator protocol and shared types.

A generator produces a :class:`Sample`: per-call independent treatment and
control arrays plus optional auxiliary structures (covariates for CUPED,
strata for post-stratification, paired numerator/denominator arrays for
ratio metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from absim.types import FloatArray

IntArray = np.ndarray[Any, np.dtype[np.integer[Any]]]


def make_strata(
    values_t: FloatArray,
    values_c: FloatArray,
    n_strata: int,
) -> tuple[IntArray, IntArray]:
    """Bin both arms into ``n_strata`` equal-frequency buckets using pooled quantiles.

    Returns integer stratum labels in ``[0, n_strata)`` for each arm. When
    ``n_strata <= 1`` returns all-zero arrays — the canonical "no stratification"
    fallback that keeps downstream code uniform.
    """
    n_strata = max(1, int(n_strata))
    if n_strata == 1:
        return np.zeros(values_t.size, dtype=int), np.zeros(values_c.size, dtype=int)
    edges = np.quantile(np.concatenate([values_t, values_c]), np.linspace(0, 1, n_strata + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.digitize(values_t, edges[1:-1]), np.digitize(values_c, edges[1:-1])


@dataclass(frozen=True, slots=True)
class Sample:
    """A single Monte Carlo sample for one (data, criterion) iteration.

    Attributes
    ----------
    treatment, control
        Per-unit outcomes :math:`Y_i`. For ratio metrics this is the realised
        :math:`N_i / D_i` (criteria that ignore it can simply leave it).
    aux
        Free-form auxiliary arrays — generators that produce covariates,
        strata, numerator/denominator pairs, or features for CUPAC place
        them here using the **same kwarg names** the corresponding criteria
        consume (e.g. ``"covariate_treatment"``, ``"strata_control"``,
        ``"numerator_treatment"``).
    """

    treatment: FloatArray
    control: FloatArray
    aux: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Generator(Protocol):
    """Protocol for synthetic data generators."""

    name: str

    def sample(self, rng: np.random.Generator, mean_shift: float) -> Sample:
        """Draw one sample with the requested treatment-effect shift.

        ``mean_shift`` is interpreted on the metric's natural scale —
        the simulator passes ``0`` for FPR runs and ``effect.value`` otherwise.
        Generators are expected to be deterministic given the supplied
        :class:`numpy.random.Generator`.
        """
        ...
