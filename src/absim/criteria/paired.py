"""Paired stratification — matched-pairs t-test.

The criterion supports two pairing modes:

1. **Index pairing** (preferred). When the generator emits genuinely paired
   samples (treatment and control share a within-pair latent covariate),
   pass ``paired=True`` and the criterion just takes
   :math:`d_i = Y^T_i - Y^C_i`. The differences are i.i.d., the standard
   error is correctly estimated, and the FPR is calibrated.

2. **Rank-matching fallback**. When the data is unpaired and a covariate is
   provided, pairs are formed by sorting both arms by covariate rank and
   matching by position. This is the textbook "matched pair" approximation
   for independent samples, but it has a small FPR inflation because the
   resulting :math:`d_{(i)}` are positively autocorrelated through the
   shared order-statistic structure. Use index pairing whenever possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from absim._stats import degenerate_result, make_result, two_sided_t_pvalue
from absim.criteria.base import register
from absim.types import TestResult

if TYPE_CHECKING:
    from absim.types import FloatArray


def _pair_by_covariate(
    treatment: FloatArray,
    control: FloatArray,
    cov_t: FloatArray,
    cov_c: FloatArray,
) -> FloatArray:
    """Return per-pair differences after matching units by covariate rank."""
    n = min(treatment.size, control.size)
    if n < 2:
        raise ValueError("paired test requires at least 2 units per arm")
    order_t = np.argsort(cov_t[:n], kind="stable")
    order_c = np.argsort(cov_c[:n], kind="stable")
    return treatment[order_t] - control[order_c]


@register("paired_stratification")
@dataclass(frozen=True, slots=True)
class PairedStratification:
    """Matched-pairs t-test on within-pair differences.

    Parameters
    ----------
    alpha
        Significance level.
    paired
        If ``True`` (default), ``treatment[i]`` and ``control[i]`` are
        assumed to be a real pair (e.g. from a paired generator) — the test
        uses ``d_i = T_i - C_i`` directly. If ``False``, units are paired
        post-hoc by sorting both arms by the supplied covariate rank.
    """

    alpha: float = 0.05
    paired: bool = True
    name: str = "paired_stratification"

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run a paired t-test on within-pair differences."""
        diffs: FloatArray
        if self.paired:
            n = min(treatment.size, control.size)
            if n < 2:
                raise ValueError("paired test requires at least 2 units per arm")
            diffs = treatment[:n] - control[:n]
        else:
            try:
                cov_t = np.asarray(kwargs["covariate_treatment"], dtype=float)
                cov_c = np.asarray(kwargs["covariate_control"], dtype=float)
            except KeyError as exc:
                raise ValueError(
                    "PairedStratification(paired=False) requires "
                    "`covariate_treatment` and `covariate_control` kwargs"
                ) from exc
            diffs = _pair_by_covariate(treatment, control, cov_t, cov_c)
        n = diffs.size
        mean = float(np.mean(diffs))
        var = float(np.var(diffs, ddof=1))
        se = float(np.sqrt(var / n))
        if se == 0.0:
            return degenerate_result(mean, metadata={"n_pairs": n})
        t = mean / se
        df = float(n - 1)
        return make_result(
            p_value=two_sided_t_pvalue(t, df),
            statistic=float(t),
            effect=mean,
            std_error=se,
            alpha=self.alpha,
            df=df,
            metadata={"n_pairs": n, "df": df},
        )
