"""Binary-metric generator (Bernoulli outcomes).

Produces 0/1 outcomes for both arms and a continuous pre-experiment covariate
that is correlated (via a logistic link) with the outcome. The covariate is
also discretised into strata for ``PostStratification``.

The point-biserial correlation between Y and X is solved numerically at
construction time via Brent's method, so the realised ``corr(Y, X)`` matches
the requested ``rho`` to ≈ 1e-3 — not the loose ``corr ≈ rho/2`` you get
from the textbook ``beta = 2·rho`` heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from absim.generators.base import Sample, make_strata

if TYPE_CHECKING:
    from absim.types import FloatArray

_CALIBRATION_SEED = 20240101
_CALIBRATION_N = 50_000


def _logistic(z: FloatArray) -> FloatArray:
    """Numerically stable logistic / sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
    return out


def _solve_intercept(x: FloatArray, beta: float, target_mean: float) -> float:
    """Bisect intercept so ``mean(sigmoid(α + β·x)) ≈ target_mean``."""
    lo, hi = -20.0, 20.0
    target = float(np.clip(target_mean, 1e-9, 1.0 - 1e-9))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _logistic(mid + beta * x).mean() > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _calibrate_beta(p_baseline: float, rho_target: float) -> float:
    """Solve for the slope ``β`` such that ``corr(Y, X) ≈ rho_target``.

    Logic
    -----
    For ``X ~ N(0, 1)`` and ``Y | X ~ Bernoulli(σ(α + β·X))``, the Pearson
    correlation ``corr(Y, X) = E[Y·X] / √(p·(1 − p))``. We approximate
    ``E[Y·X]`` by ``E[σ(α + β·X)·X]`` on a large fixed sample of ``X``,
    pick ``α`` so the empirical mean is exactly ``p_baseline``, and solve
    the resulting 1-D equation in ``β`` with Brent's method.
    """
    if abs(rho_target) < 1e-9 or p_baseline <= 0.0 or p_baseline >= 1.0:
        return 0.0
    cal_rng = np.random.default_rng(_CALIBRATION_SEED)
    x = cal_rng.standard_normal(_CALIBRATION_N)
    var_y_root = float(np.sqrt(p_baseline * (1.0 - p_baseline)))
    if var_y_root == 0.0:
        return 0.0

    def realised_corr_minus_target(beta: float) -> float:
        intercept = _solve_intercept(x, beta, p_baseline)
        probs = _logistic(intercept + beta * x)
        return float((probs * x).mean()) / var_y_root - rho_target

    sign = 1.0 if rho_target > 0 else -1.0
    upper = 30.0
    upper_value = realised_corr_minus_target(sign * upper)
    if upper_value * sign < 0.0:
        # The achievable correlation saturates below the request — return the
        # cap so the resulting realised rho is the maximum possible.
        return sign * upper
    try:
        return float(brentq(realised_corr_minus_target, 0.0, sign * upper, xtol=1e-4))
    except (RuntimeError, ValueError):
        return sign * upper


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
        Target Pearson correlation between outcome ``Y`` and covariate ``X``.
        Realised within ≈ 1e-3 by numerical calibration of ``β`` (subject to
        the maximum correlation a logistic link can produce at the requested
        ``p``; saturates gracefully when asked for the unreachable).
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
    _beta: float = field(init=False, repr=False, compare=False, default=0.0)

    def __post_init__(self) -> None:
        beta = _calibrate_beta(self.p, self.rho)
        object.__setattr__(self, "_beta", beta)

    def _draw(
        self, rng: np.random.Generator, n: int, p_eff: float
    ) -> tuple[FloatArray, FloatArray]:
        """Draw ``(y_binary, x_continuous)`` such that ``corr(y, x) ≈ rho``."""
        x = rng.standard_normal(n)
        # Beta is calibrated once at construction time; intercept is solved
        # per-draw so the *empirical* mean lands on p_eff (treatment may have
        # a shifted target probability).
        intercept = _solve_intercept(x, self._beta, p_eff)
        probs = _logistic(intercept + self._beta * x)
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
