"""Generator protocol and shared types.

A generator produces a :class:`Sample`: per-call independent treatment and
control arrays plus optional auxiliary structures (covariates for CUPED,
strata for post-stratification, paired numerator/denominator arrays for
ratio metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from absim.types import FloatArray


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
