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

from absim._stats import degenerate_result, make_result, two_sided_normal_pvalue
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
            return degenerate_result(effect)
        z = effect / se_pooled
        return make_result(
            p_value=two_sided_normal_pvalue(z),
            statistic=float(z),
            effect=effect,
            std_error=se_unpooled,
            alpha=self.alpha,
            metadata={"p_pool": p_pool},
        )
