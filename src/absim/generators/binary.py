"""Binary-metric generator (Bernoulli outcomes).

Produces 0/1 outcomes for both arms and a continuous pre-experiment covariate
that is correlated (via a logistic link) with the outcome. The covariate is
also discretised into strata for ``PostStratification``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from absim.generators.base import Sample, make_strata

if TYPE_CHECKING:
    from absim.types import FloatArray


def _logistic(z: FloatArray) -> FloatArray:
    """Numerically stable logistic / sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
    return out


@dataclass(frozen=True, slots=True)
class BinaryGenerator:
    """Generator for binary outcomes (proportions / conversion rates).

    Parameters
    ----------
    n_per_group
        Sample size per arm.
    p
        Baseline probability under control.
    rho
        Approximate correlation between outcome and covariate. Realised as a
        logistic regression on a Gaussian latent variable.
    n_strata
        Number of strata to emit as auxiliary data.
    name
        Free-form label.
    """

    name: str = "binary"
    n_per_group: int = 1000
    p: float = 0.1
    rho: float = 0.3
    n_strata: int = 4

    def _draw(
        self, rng: np.random.Generator, n: int, p_eff: float
    ) -> tuple[FloatArray, FloatArray]:
        """Draw ``(y_binary, x_continuous)`` such that ``corr(y, x) ≈ rho``."""
        x = rng.standard_normal(n)
        # Choose intercept so that mean(P(Y=1)) ≈ p_eff.
        # Approximate: pick beta from rho via a logistic-link rule of thumb,
        # then solve for the intercept by quantile matching.
        beta = float(np.clip(self.rho, -0.95, 0.95)) * 2.0
        # Find intercept such that the empirical mean ≈ p_eff.
        # Use a simple bisection on this draw — cheap and exact-enough.
        target = float(np.clip(p_eff, 1e-6, 1.0 - 1e-6))
        lo, hi = -10.0, 10.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _logistic(mid + beta * x).mean() > target:
                hi = mid
            else:
                lo = mid
        intercept = 0.5 * (lo + hi)
        probs = _logistic(intercept + beta * x)
        y = (rng.uniform(size=n) < probs).astype(float)
        return y, x

    def sample(self, rng: np.random.Generator, mean_shift: float) -> Sample:
        """Draw one Monte Carlo sample."""
        p_t = float(np.clip(self.p + mean_shift, 1e-6, 1.0 - 1e-6))
        y_t, x_t = self._draw(rng, self.n_per_group, p_t)
        y_c, x_c = self._draw(rng, self.n_per_group, self.p)
        strata_t, strata_c = make_strata(x_t, x_c, self.n_strata)
        aux = {
            "covariate_treatment": x_t,
            "covariate_control": x_c,
            "features_treatment": x_t.reshape(-1, 1),
            "features_control": x_c.reshape(-1, 1),
            "strata_treatment": strata_t,
            "strata_control": strata_c,
        }
        return Sample(treatment=y_t, control=y_c, aux=aux)
