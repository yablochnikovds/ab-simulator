r"""Delta-method test for ratio metrics.

For per-unit numerators :math:`N_i` and denominators :math:`D_i`, the metric
:math:`\\theta = \\bar N / \\bar D` has, by Taylor expansion,

.. math::
    \\widehat{\\mathrm{Var}}(\\hat\\theta) \\approx
        \\frac{1}{\\bar D^2} \\widehat{\\mathrm{Var}}(\\bar N)
        - \\frac{2 \\bar N}{\\bar D^3}
            \\widehat{\\mathrm{Cov}}(\\bar N, \\bar D)
        + \\frac{\\bar N^2}{\\bar D^4}
            \\widehat{\\mathrm{Var}}(\\bar D).

The test statistic for :math:`\\hat\\theta_T - \\hat\\theta_C` is the
difference divided by the square root of the sum of group variances, then
referenced to a standard normal.

References
----------
Hájek, J., Šidák, Z., & Sen, P. K. (1999). *Theory of Rank Tests* (delta
method exposition). See also Deng et al. (2018) "Applying the Delta Method
in Metric Analytics." KDD.
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


def _ratio_mean_variance(num: FloatArray, den: FloatArray) -> tuple[float, float]:
    r"""Return :math:`\\bar N / \\bar D` and the delta-method variance of that ratio."""
    n = num.size
    if n < 2:
        raise ValueError("delta-method requires at least 2 units")
    n_bar = float(num.mean())
    d_bar = float(den.mean())
    if d_bar == 0.0:
        raise ValueError("denominator mean is zero — delta method is undefined")
    var_n = float(np.var(num, ddof=1))
    var_d = float(np.var(den, ddof=1))
    cov_nd = float(np.cov(num, den, ddof=1)[0, 1])
    var_ratio = (
        var_n / (d_bar**2) - 2.0 * n_bar / (d_bar**3) * cov_nd + (n_bar**2) / (d_bar**4) * var_d
    ) / n
    return n_bar / d_bar, max(var_ratio, 0.0)


@register("delta_method")
@dataclass(frozen=True, slots=True)
class DeltaMethod:
    """Delta-method z-test for the difference of two ratio metrics.

    Required kwargs
    ---------------
    ``numerator_treatment``, ``denominator_treatment`` and the corresponding
    ``*_control`` per-unit arrays (each of length ``n_t`` / ``n_c``).
    The convention is to ignore the ``treatment`` and ``control`` positional
    args (those are kept for protocol compatibility — they're typically the
    realised per-unit ratio :math:`N_i / D_i`).
    """

    alpha: float = 0.05
    name: str = "delta_method"

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run a delta-method z-test on a difference of ratios."""
        try:
            num_t = np.asarray(kwargs["numerator_treatment"], dtype=float)
            den_t = np.asarray(kwargs["denominator_treatment"], dtype=float)
            num_c = np.asarray(kwargs["numerator_control"], dtype=float)
            den_c = np.asarray(kwargs["denominator_control"], dtype=float)
        except KeyError as exc:
            raise ValueError(
                "DeltaMethod requires `{numerator,denominator}_{treatment,control}` kwargs"
            ) from exc
        ratio_t, var_t = _ratio_mean_variance(num_t, den_t)
        ratio_c, var_c = _ratio_mean_variance(num_c, den_c)
        effect = ratio_t - ratio_c
        se = float(np.sqrt(var_t + var_c))
        if se == 0.0:
            return degenerate_result(effect)
        z = effect / se
        return make_result(
            p_value=two_sided_normal_pvalue(z),
            statistic=float(z),
            effect=effect,
            std_error=se,
            alpha=self.alpha,
            metadata={"ratio_treatment": ratio_t, "ratio_control": ratio_c},
        )
