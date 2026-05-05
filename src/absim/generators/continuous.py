"""Continuous-metric generators (Gaussian, lognormal, mixture).

All generators emit pre-experiment covariates (``covariate_*``) and strata
indicators (``strata_*``) so a single sample feeds every criterion that
expects auxiliary data — the criteria silently ignore unrecognised kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from absim.generators.base import Sample

if TYPE_CHECKING:
    from absim.types import FloatArray


def _draw_with_covariate(
    rng: np.random.Generator,
    n: int,
    mean: float,
    sd: float,
    rho: float,
    distribution: Literal["normal", "lognormal", "mixture"],
) -> tuple[FloatArray, FloatArray]:
    """Draw ``(Y, X)`` with corr(Y, X) ≈ ``rho``.

    ``X`` is always standard normal; ``Y`` is built from a noise term and
    correlated component of ``X``, then optionally transformed.
    """
    rho = float(np.clip(rho, -0.999, 0.999))
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    z = rho * x + np.sqrt(1.0 - rho * rho) * eps
    if distribution == "normal":
        y = mean + sd * z
    elif distribution == "lognormal":
        y = np.exp(mean + sd * z)
    elif distribution == "mixture":
        # 90% N(mean, sd), 10% N(mean + 5*sd, sd) — heavy right tail.
        bump = rng.binomial(1, 0.1, size=n).astype(float)
        y = mean + sd * z + 5.0 * sd * bump
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown distribution {distribution!r}")
    return y.astype(float), x.astype(float)


@dataclass(frozen=True, slots=True)
class ContinuousGenerator:
    """Generator for continuous outcomes with optional pre-experiment covariate.

    Parameters
    ----------
    n_per_group
        Sample size per arm.
    mean
        Baseline mean of the outcome (control).
    sd
        Standard deviation of the outcome's noise term.
    rho
        Correlation between outcome and pre-experiment covariate. Higher
        ``|rho|`` ⇒ more variance reduction available to CUPED / paired tests.
    distribution
        Noise model: Gaussian, lognormal, or a heavy-tailed mixture.
    n_strata
        Number of equal-probability strata indicators emitted as auxiliary
        data (used by ``PostStratification``). Set to ``1`` to disable.
    paired
        When ``True``, treatment and control are drawn jointly: pair ``i``
        shares a latent covariate but has independent noise terms. This is
        the canonical setup for matched-pair designs (cluster sampling,
        before/after measurements, ...). When ``False`` (default) the two
        arms are fully independent, matching a standard A/B test.
        Marginal distributions are identical in both modes; only the within-
        pair correlation changes. Use ``paired=True`` to validate
        ``PairedStratification``.
    name
        Free-form label, propagated into reports.
    """

    name: str = "continuous"
    n_per_group: int = 500
    mean: float = 0.0
    sd: float = 1.0
    rho: float = 0.5
    distribution: Literal["normal", "lognormal", "mixture"] = "normal"
    n_strata: int = 4
    paired: bool = False

    def sample(self, rng: np.random.Generator, mean_shift: float) -> Sample:
        """Draw one Monte Carlo sample."""
        if self.paired:
            # Joint draw: pairs share a covariate; noise differs per arm.
            x = rng.standard_normal(self.n_per_group)
            eps_t = rng.standard_normal(self.n_per_group)
            eps_c = rng.standard_normal(self.n_per_group)
            rho = float(np.clip(self.rho, -0.999, 0.999))
            z_t = rho * x + np.sqrt(1.0 - rho * rho) * eps_t
            z_c = rho * x + np.sqrt(1.0 - rho * rho) * eps_c
            if self.distribution == "normal":
                y_t = self.mean + mean_shift + self.sd * z_t
                y_c = self.mean + self.sd * z_c
            elif self.distribution == "lognormal":
                y_t = np.exp(self.mean + mean_shift + self.sd * z_t)
                y_c = np.exp(self.mean + self.sd * z_c)
            else:  # mixture
                bump_t = rng.binomial(1, 0.1, size=self.n_per_group).astype(float)
                bump_c = rng.binomial(1, 0.1, size=self.n_per_group).astype(float)
                y_t = self.mean + mean_shift + self.sd * z_t + 5.0 * self.sd * bump_t
                y_c = self.mean + self.sd * z_c + 5.0 * self.sd * bump_c
            x_t = x.astype(float)
            x_c = x.astype(float)
            y_t = y_t.astype(float)
            y_c = y_c.astype(float)
        else:
            y_t, x_t = _draw_with_covariate(
                rng, self.n_per_group, self.mean + mean_shift, self.sd, self.rho, self.distribution
            )
            y_c, x_c = _draw_with_covariate(
                rng, self.n_per_group, self.mean, self.sd, self.rho, self.distribution
            )
        # Strata: discretise covariate into K equal-width buckets.
        n_strata = max(1, int(self.n_strata))
        if n_strata == 1:
            strata_t: np.ndarray[Any, np.dtype[np.integer[Any]]] = np.zeros(
                self.n_per_group, dtype=int
            )
            strata_c: np.ndarray[Any, np.dtype[np.integer[Any]]] = np.zeros(
                self.n_per_group, dtype=int
            )
        else:
            edges = np.quantile(np.concatenate([x_t, x_c]), np.linspace(0, 1, n_strata + 1))
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            strata_t = np.digitize(x_t, edges[1:-1])
            strata_c = np.digitize(x_c, edges[1:-1])
        aux = {
            "covariate_treatment": x_t,
            "covariate_control": x_c,
            "features_treatment": x_t.reshape(-1, 1),
            "features_control": x_c.reshape(-1, 1),
            "strata_treatment": strata_t,
            "strata_control": strata_c,
        }
        return Sample(treatment=y_t, control=y_c, aux=aux)
