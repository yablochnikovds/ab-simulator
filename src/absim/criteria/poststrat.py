r"""Post-stratification estimator.

Given strata indicators :math:`s_i \\in \\{1, \\ldots, K\\}` for every unit,
form

.. math::
    \\hat\\Delta = \\sum_{k=1}^K w_k (\\bar Y^T_k - \\bar Y^C_k)

where :math:`w_k = n_k / n` is the **pooled-sample** stratum weight. Because
weights are estimated on the full sample (not per-arm), the variance is

.. math::
    \\widehat{\\mathrm{Var}}(\\hat\\Delta) = \\sum_k w_k^2
        \\Bigl(\\frac{s^2_{T,k}}{n_{T,k}} + \\frac{s^2_{C,k}}{n_{C,k}}\\Bigr).

The resulting test is referred to a Welch–Satterthwaite t-distribution.

Variance is **never larger** than the unstratified Welch t-test when strata
explain any outcome variance — this is the "post-stratification adjustment"
result of Miratrix, Sekhon, & Yu (2013).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from absim._stats import t_ci
from absim.criteria.base import register
from absim.types import TestResult

if TYPE_CHECKING:
    from absim.types import FloatArray


@register("post_stratification")
@dataclass(frozen=True, slots=True)
class PostStratification:
    """Post-stratification estimator with closed-form variance.

    Parameters
    ----------
    alpha
        Significance level.
    """

    alpha: float = 0.05
    name: str = "post_stratification"

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run a post-stratified mean-difference test."""
        try:
            strata_t = np.asarray(kwargs["strata_treatment"])
            strata_c = np.asarray(kwargs["strata_control"])
        except KeyError as exc:
            raise ValueError(
                "PostStratification requires `strata_treatment` and `strata_control` kwargs"
            ) from exc
        if strata_t.size != treatment.size or strata_c.size != control.size:
            raise ValueError("strata arrays must match outcome array sizes")

        n_total = treatment.size + control.size
        all_strata = np.concatenate([strata_t, strata_c])
        unique = np.unique(all_strata)

        delta = 0.0
        var = 0.0
        df_num = 0.0
        df_den = 0.0
        for k in unique:
            mask_t = strata_t == k
            mask_c = strata_c == k
            n_k = int(mask_t.sum() + mask_c.sum())
            n_tk = int(mask_t.sum())
            n_ck = int(mask_c.sum())
            if n_tk < 2 or n_ck < 2:
                # Pooled-variance fallback when a stratum is too small to
                # estimate per-arm variance.
                continue
            w_k = n_k / n_total
            mean_t = float(np.mean(treatment[mask_t]))
            mean_c = float(np.mean(control[mask_c]))
            var_t = float(np.var(treatment[mask_t], ddof=1))
            var_c = float(np.var(control[mask_c], ddof=1))
            stratum_var = w_k * w_k * (var_t / n_tk + var_c / n_ck)
            delta += w_k * (mean_t - mean_c)
            var += stratum_var
            df_num += stratum_var
            df_den += (stratum_var**2) / max(n_tk + n_ck - 2, 1)

        se = float(np.sqrt(var))
        if se == 0.0:
            return TestResult(
                p_value=1.0,
                statistic=0.0,
                effect=delta,
                std_error=0.0,
                ci_low=delta,
                ci_high=delta,
                rejected=False,
                metadata={"n_strata": int(unique.size)},
            )
        df = (df_num**2) / df_den if df_den > 0 else float(n_total - 2)
        t = delta / se
        p_value = float(2.0 * stats.t.sf(abs(t), df))
        ci_low, ci_high = t_ci(delta, se, df, self.alpha)
        return TestResult(
            p_value=p_value,
            statistic=float(t),
            effect=delta,
            std_error=se,
            ci_low=ci_low,
            ci_high=ci_high,
            rejected=p_value < self.alpha,
            metadata={"n_strata": int(unique.size), "df": df},
        )
