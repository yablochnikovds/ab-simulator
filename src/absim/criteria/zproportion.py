r"""Two-sample z-test for proportions (binary metrics).

Test statistic
--------------
.. math::
    z = \\frac{\\hat p_T - \\hat p_C}
              {\\sqrt{\\bar p (1 - \\bar p) (1/n_T + 1/n_C)}}

where :math:`\\bar p` is the pooled proportion under H₀. The CI uses an
unpooled standard error:

.. math::
    \\mathrm{SE}_{\\text{CI}} = \\sqrt{\\hat p_T (1 - \\hat p_T)/n_T +
                                       \\hat p_C (1 - \\hat p_C)/n_C}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from absim._stats import normal_ci, two_sided_normal_pvalue
from absim.criteria.base import register
from absim.types import TestResult

if TYPE_CHECKING:
    from absim.types import FloatArray


@register("z_proportion")
@dataclass(frozen=True, slots=True)
class ZTestProportions:
    """Two-sample z-test for proportions.

    Parameters
    ----------
    alpha
        Significance level. Used for the unpooled-SE CI and rejection rule.
    """

    alpha: float = 0.05
    name: str = "z_proportion"

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run a two-sample z-test on 0/1 data."""
        n_t, n_c = treatment.size, control.size
        if n_t < 1 or n_c < 1:
            raise ValueError("z_proportion requires at least 1 observation per group")
        p_t = float(np.mean(treatment))
        p_c = float(np.mean(control))
        effect = p_t - p_c
        # Pooled SE for the test statistic (under H0).
        p_pool = (p_t * n_t + p_c * n_c) / (n_t + n_c)
        se_pooled = float(np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_t + 1.0 / n_c)))
        # Unpooled SE for the CI.
        se_unpooled = float(np.sqrt(p_t * (1.0 - p_t) / n_t + p_c * (1.0 - p_c) / n_c))
        if se_pooled == 0.0:
            return TestResult(
                p_value=1.0,
                statistic=0.0,
                effect=effect,
                std_error=0.0,
                ci_low=effect,
                ci_high=effect,
                rejected=False,
            )
        z = effect / se_pooled
        p_value = two_sided_normal_pvalue(z)
        ci_low, ci_high = normal_ci(effect, se_unpooled, self.alpha)
        return TestResult(
            p_value=p_value,
            statistic=float(z),
            effect=effect,
            std_error=se_unpooled,
            ci_low=ci_low,
            ci_high=ci_high,
            rejected=p_value < self.alpha,
            metadata={"p_pool": p_pool},
        )
