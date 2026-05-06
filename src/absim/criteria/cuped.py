r"""CUPED — Controlled-experiment Using Pre-Experiment Data.

Variance reduction via pre-experiment covariate :math:`X`. Define

.. math::
    \\theta = \\frac{\\mathrm{Cov}(Y, X)}{\\mathrm{Var}(X)},
    \\quad
    Y_{\\mathrm{adj}} = Y - \\theta \\bigl(X - \\mathbb E X \\bigr),

then run Welch's t-test on :math:`Y_{\\mathrm{adj}}`. Variance reduction is
:math:`1 - \\rho^2` where :math:`\\rho = \\mathrm{Corr}(Y, X)`.

References
----------
Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity
of online controlled experiments by utilizing pre-experiment data. *WSDM '13*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from absim._stats import make_result, welch_ttest
from absim.criteria.base import register
from absim.types import TestResult

if TYPE_CHECKING:
    from absim.types import FloatArray


@register("cuped")
@dataclass(frozen=True, slots=True)
class CUPED:
    """CUPED-adjusted Welch t-test.

    The covariate must be supplied per-call via the ``covariate_treatment``
    and ``covariate_control`` keyword arguments (typically a per-unit
    pre-experiment metric).

    Parameters
    ----------
    alpha
        Significance level. Used for the CI and the rejection decision.
    """

    alpha: float = 0.05
    name: str = "cuped"

    @staticmethod
    def _theta(y: FloatArray, x: FloatArray) -> float:
        """Pooled OLS slope of Y on (X - mean(X))."""
        x_centered = x - x.mean()
        var_x = float(np.dot(x_centered, x_centered) / max(x.size - 1, 1))
        if var_x == 0.0:
            return 0.0
        cov_xy = float(np.dot(x_centered, y - y.mean()) / max(x.size - 1, 1))
        return cov_xy / var_x

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run a CUPED-adjusted Welch t-test."""
        try:
            cov_t: FloatArray = kwargs["covariate_treatment"]
            cov_c: FloatArray = kwargs["covariate_control"]
        except KeyError as exc:
            raise ValueError(
                "CUPED requires `covariate_treatment` and `covariate_control` kwargs"
            ) from exc
        if cov_t.size != treatment.size or cov_c.size != control.size:
            raise ValueError("covariate arrays must match outcome array sizes")
        # Pool both groups to estimate theta — standard practice because the
        # covariate is independent of treatment assignment.
        y_pool = np.concatenate([treatment, control])
        x_pool = np.concatenate([cov_t, cov_c])
        theta = self._theta(y_pool, x_pool)
        x_mean = float(x_pool.mean())
        adj_t = treatment - theta * (cov_t - x_mean)
        adj_c = control - theta * (cov_c - x_mean)
        statistic, p_value, effect, se, df = welch_ttest(adj_t, adj_c)
        return make_result(
            p_value=p_value,
            statistic=statistic,
            effect=effect,
            std_error=se,
            alpha=self.alpha,
            df=df,
            metadata={"theta": theta, "df": df},
        )
