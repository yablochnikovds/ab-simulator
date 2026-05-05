"""Core dataclasses and type aliases used throughout :mod:`absim`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating[Any]]
else:  # pragma: no cover - runtime placeholder
    FloatArray = Any


@dataclass(frozen=True, slots=True)
class TestResult:
    """Outcome of running a single statistical criterion on a single sample.

    Parameters
    ----------
    p_value
        The two-sided p-value of the test. Must lie in :math:`[0, 1]`.
    statistic
        The test statistic (z, t, U, etc.) — semantics depend on the criterion.
    effect
        The point estimate of the treatment effect (typically the mean
        difference, on the metric's natural scale).
    std_error
        The standard error of the effect estimate, when available. ``nan`` if
        the criterion cannot produce one (e.g. percentile bootstrap).
    ci_low
        Lower bound of the (1 - alpha) confidence interval for the effect.
    ci_high
        Upper bound of the (1 - alpha) confidence interval for the effect.
    rejected
        Whether the null hypothesis was rejected at the configured ``alpha``.
    metadata
        Free-form criterion-specific diagnostics (e.g. CUPED's theta).
    """

    p_value: float
    statistic: float
    effect: float
    std_error: float
    ci_low: float
    ci_high: float
    rejected: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EffectSize:
    """A specification of the effect injected into the synthetic data.

    Parameters
    ----------
    name
        A human-readable label (e.g. ``"none"``, ``"small"``, ``"medium"``).
    value
        Numeric magnitude on the metric's native scale (mean shift, lift in
        probability, multiplicative ratio shift, ...). ``0.0`` ⇒ H₀.
    relative
        Whether ``value`` is interpreted as a fractional/multiplicative shift
        (``True``) or an absolute shift (``False``).
    """

    name: str
    value: float
    relative: bool = False


@dataclass(frozen=True, slots=True)
class SimulationReport:
    """Aggregated outcome of ``N`` simulations of one (data × criterion) pair.

    The "rejection rate" estimates the **FPR** under H₀ (``effect.value == 0``)
    and the **Power** under H₁ (``effect.value != 0``).

    Attributes
    ----------
    criterion_name
        Name of the criterion used.
    n_sims
        Number of Monte Carlo simulations executed.
    alpha
        Nominal significance level used to reject H₀.
    effect
        The injected effect size.
    rejection_rate
        Fraction of simulations rejecting H₀. Equals FPR when H₀ is true,
        Power when H₁ is true.
    binomial_ci_low, binomial_ci_high
        Wilson 95% confidence interval for ``rejection_rate``.
    mean_pvalue
        Mean of all simulation p-values.
    mean_effect
        Mean of all simulation point estimates.
    mean_std_error
        Mean of all simulation standard errors. ``nan`` if criterion has none.
    runtime_sec
        Wall-clock time of the simulation, in seconds.
    metadata
        Free-form simulator-level diagnostics.
    """

    criterion_name: str
    n_sims: int
    alpha: float
    effect: EffectSize
    rejection_rate: float
    binomial_ci_low: float
    binomial_ci_high: float
    mean_pvalue: float
    mean_effect: float
    mean_std_error: float
    runtime_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fpr(self) -> float:
        """Alias for :attr:`rejection_rate` when running under H₀."""
        return self.rejection_rate

    @property
    def power(self) -> float:
        """Alias for :attr:`rejection_rate` when running under H₁."""
        return self.rejection_rate

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for tabular reporting / parquet IO."""
        return {
            "criterion": self.criterion_name,
            "n_sims": self.n_sims,
            "alpha": self.alpha,
            "effect_name": self.effect.name,
            "effect_value": self.effect.value,
            "effect_relative": self.effect.relative,
            "rejection_rate": self.rejection_rate,
            "ci_low": self.binomial_ci_low,
            "ci_high": self.binomial_ci_high,
            "mean_pvalue": self.mean_pvalue,
            "mean_effect": self.mean_effect,
            "mean_std_error": self.mean_std_error,
            "runtime_sec": self.runtime_sec,
        }
