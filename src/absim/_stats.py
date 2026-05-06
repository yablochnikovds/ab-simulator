"""Numerical helpers shared across criteria.

Everything here uses :class:`numpy.random.Generator` (no legacy ``np.random``).
Functions are deterministic given identical inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from absim.types import TestResult

if TYPE_CHECKING:
    from absim.types import FloatArray


def welch_ttest(
    treatment: FloatArray,
    control: FloatArray,
) -> tuple[float, float, float, float, float]:
    """Welch's two-sample t-test (unequal variances, two-sided).

    Parameters
    ----------
    treatment, control
        1-D arrays of observations for the two groups.

    Returns
    -------
    statistic, p_value, effect, std_error, df
        ``effect = mean(treatment) - mean(control)``, ``std_error`` is the SE
        of that difference, ``df`` is the Welch–Satterthwaite degrees of freedom
        (returned even when ``std_error == 0`` so callers can build a CI).
    """
    if treatment.size < 2 or control.size < 2:
        raise ValueError("welch_ttest requires at least 2 observations per group")
    mean_t = float(np.mean(treatment))
    mean_c = float(np.mean(control))
    var_t = float(np.var(treatment, ddof=1))
    var_c = float(np.var(control, ddof=1))
    n_t, n_c = treatment.size, control.size
    se = float(np.sqrt(var_t / n_t + var_c / n_c))
    effect = mean_t - mean_c
    df_num = (var_t / n_t + var_c / n_c) ** 2
    df_den = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    df = df_num / df_den if df_den > 0 else float(n_t + n_c - 2)
    if se == 0.0:
        return 0.0, 1.0, effect, 0.0, df
    t = effect / se
    return float(t), two_sided_t_pvalue(t, df), effect, se, df


def two_sided_normal_pvalue(z: float) -> float:
    """Two-sided p-value of a standard-normal test statistic."""
    return float(2.0 * stats.norm.sf(abs(z)))


def two_sided_t_pvalue(t: float, df: float) -> float:
    """Two-sided p-value of a Student-t test statistic with ``df`` d.o.f."""
    return float(2.0 * stats.t.sf(abs(t), df))


def normal_ci(estimate: float, std_error: float, alpha: float) -> tuple[float, float]:
    """Two-sided normal-approximation confidence interval."""
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return estimate - z * std_error, estimate + z * std_error


def t_ci(estimate: float, std_error: float, df: float, alpha: float) -> tuple[float, float]:
    """Two-sided Student-t confidence interval."""
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return estimate - crit * std_error, estimate + crit * std_error


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    More accurate than the normal approximation, especially near 0 / 1 and for
    small ``n``. Used to put a confidence band around the simulator's
    estimated rejection rate.
    """
    if n <= 0:
        return 0.0, 0.0
    p_hat = successes / n
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))) / denom
    low = max(0.0, centre - half)
    high = min(1.0, centre + half)
    return float(low), float(high)


def split_seed(seed: int | np.random.SeedSequence, n: int) -> list[np.random.SeedSequence]:
    """Split a root seed into ``n`` independent child :class:`SeedSequence` objects.

    Used to give every Monte Carlo iteration its own statistically independent
    stream while keeping the run reproducible from a single integer seed.
    """
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return ss.spawn(n)


def make_rng(seed: int | np.random.SeedSequence | np.random.Generator) -> np.random.Generator:
    """Coerce a seed-like input into a fresh :class:`numpy.random.Generator`."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def degenerate_result(effect: float, *, metadata: dict[str, Any] | None = None) -> TestResult:
    """Return a no-signal :class:`TestResult` (SE = 0, p = 1, fail to reject).

    Used when the data has insufficient variance to compute a real test statistic
    (e.g. constant arrays, all-equal samples, degenerate strata).
    """
    return TestResult(
        p_value=1.0,
        statistic=0.0,
        effect=effect,
        std_error=0.0,
        ci_low=effect,
        ci_high=effect,
        rejected=False,
        metadata=metadata or {},
    )


def make_result(
    *,
    p_value: float,
    statistic: float,
    effect: float,
    std_error: float,
    alpha: float,
    df: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> TestResult:
    """Assemble a :class:`TestResult` with the appropriate CI flavour.

    Picks Student-t CI when ``df`` is given, normal CI otherwise. Collapses the
    SE-equals-zero case to a degenerate (effect, effect) interval.
    """
    if std_error <= 0.0:
        ci_low = ci_high = effect
    elif df is None:
        ci_low, ci_high = normal_ci(effect, std_error, alpha)
    else:
        ci_low, ci_high = t_ci(effect, std_error, df, alpha)
    return TestResult(
        p_value=p_value,
        statistic=statistic,
        effect=effect,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        rejected=p_value < alpha,
        metadata=metadata or {},
    )
