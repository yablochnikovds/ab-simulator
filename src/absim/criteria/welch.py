"""Welch's two-sample t-test.

References
----------
Welch, B. L. (1947). The generalization of "Student's" problem when several
different population variances are involved. *Biometrika*, 34(1/2), 28–35.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from absim._stats import make_result, welch_ttest
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
        statistic, p_value, effect, se, df = welch_ttest(treatment, control)
        return make_result(
            p_value=p_value,
            statistic=statistic,
            effect=effect,
            std_error=se,
            alpha=self.alpha,
            df=df,
            metadata={"df": df},
        )
