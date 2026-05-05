r"""Budylin linearization for ratio metrics.

Reduce the ratio :math:`\\bar N / \\bar D` to an *additive* per-unit metric

.. math::
    L_i = \\frac{N_i}{\\bar D} - \\frac{\\bar N}{\\bar D^2} D_i,

then run any sample-mean test (here Welch's t-test). To leading order this
is identical to the delta method, but lets us drop in stratified variance
reduction techniques (CUPED, post-stratification, ...) on top.

The linearization is computed using the **pooled** sample means of
:math:`N` and :math:`D` (Budylin's recipe), so both arms share the same
linear functional and the comparison is unbiased.

References
----------
Budylin, R. (2018). "Consistent transformation of ratio metrics for efficient
online controlled experiments."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from absim._stats import t_ci, welch_ttest
from absim.criteria.base import register
from absim.types import TestResult

if TYPE_CHECKING:
    from absim.types import FloatArray


def _linearize(
    num_t: FloatArray, den_t: FloatArray, num_c: FloatArray, den_c: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Linearise per-unit (N, D) into a single additive metric per unit.

    Both arms are linearised against the **pooled** numerator and
    denominator means.
    """
    n_pool = np.concatenate([num_t, num_c])
    d_pool = np.concatenate([den_t, den_c])
    n_bar = float(n_pool.mean())
    d_bar = float(d_pool.mean())
    if d_bar == 0.0:
        raise ValueError("pooled denominator mean is zero — cannot linearise")
    coef = n_bar / (d_bar**2)
    L_t = num_t / d_bar - coef * den_t
    L_c = num_c / d_bar - coef * den_c
    return L_t, L_c


@register("linearization")
@dataclass(frozen=True, slots=True)
class Linearization:
    """Budylin linearization + Welch's t-test on the linearised metric.

    Required kwargs
    ---------------
    ``numerator_treatment``, ``denominator_treatment``,
    ``numerator_control``, ``denominator_control``.
    """

    alpha: float = 0.05
    name: str = "linearization"

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run a Welch t-test on the Budylin-linearised ratio metric."""
        try:
            num_t = np.asarray(kwargs["numerator_treatment"], dtype=float)
            den_t = np.asarray(kwargs["denominator_treatment"], dtype=float)
            num_c = np.asarray(kwargs["numerator_control"], dtype=float)
            den_c = np.asarray(kwargs["denominator_control"], dtype=float)
        except KeyError as exc:
            raise ValueError(
                "Linearization requires `{numerator,denominator}_{treatment,control}` kwargs"
            ) from exc
        L_t, L_c = _linearize(num_t, den_t, num_c, den_c)
        statistic, p_value, effect, se = welch_ttest(L_t, L_c)
        n_t, n_c = L_t.size, L_c.size
        var_t = float(np.var(L_t, ddof=1))
        var_c = float(np.var(L_c, ddof=1))
        df_num = (var_t / n_t + var_c / n_c) ** 2
        df_den = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
        df = df_num / df_den if df_den > 0 else float(n_t + n_c - 2)
        ci_low, ci_high = t_ci(effect, se, df, self.alpha) if se > 0 else (effect, effect)
        return TestResult(
            p_value=p_value,
            statistic=statistic,
            effect=effect,
            std_error=se,
            ci_low=ci_low,
            ci_high=ci_high,
            rejected=p_value < self.alpha,
            metadata={"df": df},
        )
