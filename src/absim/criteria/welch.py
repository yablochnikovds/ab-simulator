"""Welch's two-sample t-test.

References
----------
Welch, B. L. (1947). The generalization of "Student's" problem when several
different population variances are involved. *Biometrika*, 34(1/2), 28–35.
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


@register("welch_t")
@dataclass(frozen=True, slots=True)
class WelchTTest:
    r"""Welch's t-test for the difference of two means with unequal variances.

    Parameters
    ----------
    alpha
        Significance level used to construct the CI and decide rejection.
        Note that the simulator may also override this when collecting results.

    Notes
    -----
    The test statistic is

    .. math::
        t = \\frac{\\bar{Y}_T - \\bar{Y}_C}
                  {\\sqrt{s_T^2/n_T + s_C^2/n_C}}

    with Welch–Satterthwaite degrees of freedom

    .. math::
        \\nu = \\frac{(s_T^2/n_T + s_C^2/n_C)^2}
                     {(s_T^2/n_T)^2/(n_T-1) + (s_C^2/n_C)^2/(n_C-1)}.
    """

    alpha: float = 0.05
    name: str = "welch_t"

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run Welch's t-test and return a :class:`TestResult`."""
        statistic, p_value, effect, se = welch_ttest(treatment, control)
        n_t, n_c = treatment.size, control.size
        # Approximate Welch–Satterthwaite df for CI.
        var_t = float(np.var(treatment, ddof=1))
        var_c = float(np.var(control, ddof=1))
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
