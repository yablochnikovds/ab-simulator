r"""Bootstrap criterion — non-parametric percentile and BCa intervals.

For each replicate :math:`b = 1, \\ldots, B`:

1. Resample treatment and control independently *with replacement*.
2. Compute the difference of resampled means.

The resulting bootstrap distribution gives the percentile CI directly.
The BCa (bias-corrected and accelerated) CI additionally adjusts for bias and
skewness, with the acceleration constant estimated by jackknife:

.. math::
    a = \\frac{\\sum_i (\\bar\\theta_{(\\cdot)} - \\theta_{(i)})^3}
              {6 \\bigl[\\sum_i (\\bar\\theta_{(\\cdot)} - \\theta_{(i)})^2\\bigr]^{3/2}}.

References
----------
Efron, B., & Tibshirani, R. (1994). *An Introduction to the Bootstrap.*
Chapman & Hall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats

from absim.criteria.base import register
from absim.types import TestResult

if TYPE_CHECKING:
    from absim.types import FloatArray


def _resample_means(rng: np.random.Generator, x: FloatArray, n_boot: int) -> FloatArray:
    """Vectorised bootstrap means: shape (n_boot,)."""
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    out: FloatArray = x[idx].mean(axis=1)
    return out


def _bca_endpoints(
    boot_dist: FloatArray, observed: float, jack: FloatArray, alpha: float
) -> tuple[float, float]:
    """BCa endpoints for a (1 - alpha) two-sided CI."""
    z0 = float(stats.norm.ppf((boot_dist < observed).mean())) if boot_dist.size > 0 else 0.0
    jack_mean = jack.mean()
    diff = jack_mean - jack
    num = np.sum(diff**3)
    den = 6.0 * (np.sum(diff**2) ** 1.5)
    a_hat = float(num / den) if den != 0.0 else 0.0
    z_a_lo = float(stats.norm.ppf(alpha / 2.0))
    z_a_hi = float(stats.norm.ppf(1.0 - alpha / 2.0))

    def _adjust(z: float) -> float:
        denom = 1.0 - a_hat * (z0 + z)
        if denom == 0.0:
            denom = 1e-12
        return float(stats.norm.cdf(z0 + (z0 + z) / denom))

    p_lo = _adjust(z_a_lo)
    p_hi = _adjust(z_a_hi)
    lo = float(np.quantile(boot_dist, np.clip(p_lo, 0.0, 1.0)))
    hi = float(np.quantile(boot_dist, np.clip(p_hi, 0.0, 1.0)))
    return lo, hi


@register("bootstrap")
@dataclass(frozen=True, slots=True)
class Bootstrap:
    """Non-parametric bootstrap for the difference of means.

    Parameters
    ----------
    alpha
        Significance level.
    n_resamples
        Number of bootstrap replicates :math:`B`.
    method
        ``"percentile"`` or ``"bca"`` — controls which CI / p-value is reported.
    seed
        Optional integer seed; if ``None`` the criterion uses a fresh
        ``np.random.default_rng()``. The simulator typically passes the
        per-iteration seed via the ``rng`` kwarg instead.
    """

    alpha: float = 0.05
    n_resamples: int = 2000
    method: Literal["percentile", "bca"] = "percentile"
    seed: int | None = None
    name: str = "bootstrap"

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run the bootstrap and return percentile or BCa CI + p-value."""
        rng_arg = kwargs.get("rng")
        if isinstance(rng_arg, np.random.Generator):
            rng = rng_arg
        else:
            rng = np.random.default_rng(self.seed)
        observed = float(treatment.mean() - control.mean())
        boot_t = _resample_means(rng, treatment, self.n_resamples)
        boot_c = _resample_means(rng, control, self.n_resamples)
        boot_diff = boot_t - boot_c
        se = float(np.std(boot_diff, ddof=1))

        if self.method == "percentile":
            q_lo, q_hi = np.quantile(boot_diff, [self.alpha / 2.0, 1.0 - self.alpha / 2.0])
            lo, hi = float(q_lo), float(q_hi)
        else:  # bca
            jack_t = _jackknife_means(treatment)
            jack_c = _jackknife_means(control)
            jack_diff = np.concatenate([jack_t - control.mean(), treatment.mean() - jack_c])
            lo, hi = _bca_endpoints(boot_diff, observed, jack_diff, self.alpha)

        # P-value via the achieved-significance-level formulation: the
        # smallest alpha at which the CI excludes zero.
        # Using the bootstrap distribution's tail mass on the side toward zero.
        tail = float(np.minimum((boot_diff <= 0).mean(), (boot_diff >= 0).mean()))
        p_value = float(min(1.0, 2.0 * tail))
        rejected = lo > 0.0 or hi < 0.0
        return TestResult(
            p_value=p_value,
            statistic=observed / se if se > 0 else 0.0,
            effect=observed,
            std_error=se,
            ci_low=lo,
            ci_high=hi,
            rejected=rejected,
            metadata={"method": self.method, "n_resamples": self.n_resamples},
        )


def _jackknife_means(x: FloatArray) -> FloatArray:
    """Leave-one-out means for jackknife acceleration estimation."""
    n = x.size
    total = x.sum()
    out: FloatArray = (total - x) / (n - 1)
    return out
