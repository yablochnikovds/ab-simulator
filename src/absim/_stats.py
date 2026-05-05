"""Numerical helpers shared across criteria.

Everything here uses :class:`numpy.random.Generator` (no legacy ``np.random``).
Functions are deterministic given identical inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from absim.types import FloatArray


def welch_ttest(
    treatment: FloatArray,
    control: FloatArray,
) -> tuple[float, float, float, float]:
    """Welch's two-sample t-test (unequal variances, two-sided).

    Parameters
    ----------
    treatment, control
        1-D arrays of observations for the two groups.

    Returns
    -------
    statistic, p_value, effect, std_error
        ``effect = mean(treatment) - mean(control)``,
        ``std_error`` is the SE of that difference.
    """
    if treatment.size < 2 or control.size < 2:
        raise ValueError("welch_ttest requires at least 2 observations per group")
    mean_t = float(np.mean(treatment))
    mean_c = float(np.mean(control))
    var_t = float(np.var(treatment, ddof=1))
    var_c = float(np.var(control, ddof=1))
    n_t, n_c = treatment.size, control.size
    se = float(np.sqrt(var_t / n_t + var_c / n_c))
    if se == 0.0:
        # Degenerate: no variance in either group. Treat as no signal.
        return 0.0, 1.0, mean_t - mean_c, 0.0
    effect = mean_t - mean_c
    t = effect / se
    # Welch–Satterthwaite degrees of freedom.
    df_num = (var_t / n_t + var_c / n_c) ** 2
    df_den = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    df = df_num / df_den if df_den > 0 else float(n_t + n_c - 2)
    p = float(2.0 * stats.t.sf(abs(t), df))
    return float(t), p, effect, se


def two_sided_normal_pvalue(z: float) -> float:
    """Two-sided p-value of a standard-normal test statistic."""
    return float(2.0 * stats.norm.sf(abs(z)))


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
