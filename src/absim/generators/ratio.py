"""Ratio-metric generator (numerator and denominator at different granularity).

Models the canonical "clicks per session" / "purchases per visit" pattern:
each user has a *random number of sessions* :math:`D_i` (Poisson), and each
session contributes a noisy count :math:`N_i` whose mean is
``ratio * D_i``. The treatment effect is a multiplicative shift of the
underlying rate, capturing what a real lift looks like for a ratio metric.

Generators emit per-unit numerator and denominator arrays so delta-method
and linearization criteria can consume them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from absim.generators.base import Sample

if TYPE_CHECKING:
    from absim.types import FloatArray


@dataclass(frozen=True, slots=True)
class RatioGenerator:
    """Generator for ratio metrics with realistic N–D correlation.

    Parameters
    ----------
    n_per_group
        Number of users per arm.
    base_rate
        The "true" per-session ratio under control (e.g. clicks per session).
    sessions_mean
        Mean of the Poisson distribution governing per-user sessions.
        ``D_i ~ max(1, Poisson(sessions_mean))`` to avoid zero denominators.
    noise
        Additional Gaussian noise (in fraction-of-base) on the per-user rate.
    relative
        If ``True``, ``mean_shift`` is interpreted as a multiplicative lift
        (e.g. ``0.05`` ⇒ +5%); otherwise as an additive shift to the rate.
    n_strata
        Number of strata buckets emitted as auxiliary data (based on
        denominator size — a common choice for ratio metrics).
    name
        Free-form label.
    """

    name: str = "ratio"
    n_per_group: int = 1000
    base_rate: float = 0.2
    sessions_mean: float = 5.0
    noise: float = 0.1
    relative: bool = True
    n_strata: int = 4

    def _draw(
        self,
        rng: np.random.Generator,
        n: int,
        rate: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Draw ``(N, D)`` per user with realistic positive correlation."""
        d = np.maximum(1, rng.poisson(self.sessions_mean, size=n)).astype(float)
        # Per-user noisy rate (truncated to ≥ 0).
        per_user_rate = np.maximum(0.0, rate * (1.0 + self.noise * rng.standard_normal(n)))
        n_arr = rng.poisson(per_user_rate * d).astype(float)
        return n_arr, d

    def sample(self, rng: np.random.Generator, mean_shift: float) -> Sample:
        """Draw one Monte Carlo sample."""
        if self.relative:
            rate_t = self.base_rate * (1.0 + mean_shift)
        else:
            rate_t = self.base_rate + mean_shift
        rate_t = max(rate_t, 0.0)
        num_t, den_t = self._draw(rng, self.n_per_group, rate_t)
        num_c, den_c = self._draw(rng, self.n_per_group, self.base_rate)
        # Per-unit ratio for criteria that operate on ``treatment`` / ``control``
        # (e.g. Welch t-test, bootstrap). Avoid div-by-zero by using max(D, 1).
        ratio_t = num_t / np.maximum(den_t, 1.0)
        ratio_c = num_c / np.maximum(den_c, 1.0)
        # Strata bucketed by denominator size — a typical "user activity" stratifier.
        n_strata = max(1, int(self.n_strata))
        if n_strata == 1:
            strata_t: np.ndarray[Any, np.dtype[np.integer[Any]]] = np.zeros(
                self.n_per_group, dtype=int
            )
            strata_c: np.ndarray[Any, np.dtype[np.integer[Any]]] = np.zeros(
                self.n_per_group, dtype=int
            )
        else:
            edges = np.quantile(np.concatenate([den_t, den_c]), np.linspace(0, 1, n_strata + 1))
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            strata_t = np.digitize(den_t, edges[1:-1])
            strata_c = np.digitize(den_c, edges[1:-1])
        aux = {
            "numerator_treatment": num_t,
            "numerator_control": num_c,
            "denominator_treatment": den_t,
            "denominator_control": den_c,
            "covariate_treatment": den_t.astype(float),
            "covariate_control": den_c.astype(float),
            "features_treatment": den_t.reshape(-1, 1).astype(float),
            "features_control": den_c.reshape(-1, 1).astype(float),
            "strata_treatment": strata_t,
            "strata_control": strata_c,
        }
        return Sample(treatment=ratio_t, control=ratio_c, aux=aux)
