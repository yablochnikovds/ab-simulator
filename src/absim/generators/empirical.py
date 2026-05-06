"""Empirical (real-data-driven) generator.

Bootstrap-resamples from observed historical metric values so the simulator
runs on a distribution that matches your production data — including its
zero-inflation, heavy tails, multi-modality, and other quirks the parametric
generators (Gaussian / lognormal / mixture / Bernoulli / Poisson) won't
reproduce.

This closes the "but I can't run absim on my real data" objection: feed the
generator a 1-D array of historical outcomes (e.g. one-month per-user revenue
from your warehouse), optionally with paired pre-period covariates / strata /
numerator-denominator pairs, and run any criterion through 10 000+ simulated
experiments with calibrated effect injection.

Reference
---------
The empirical-bootstrap data-generating process is the standard "permutation
plus injection" recipe used by industry experimentation platforms (e.g.
Booking, Spotify, DoorDash) for power analysis on real production metrics.
See also: Politis, Romano & Wolf (1999), *Subsampling*, Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from absim.generators.base import Sample

if TYPE_CHECKING:
    from absim.types import FloatArray

_BINARY_SET = frozenset({0.0, 1.0})


@dataclass(frozen=True, slots=True)
class EmpiricalGenerator:
    """Bootstrap-based generator that uses your real data as the DGP.

    Parameters
    ----------
    outcomes
        1-D array of observed metric values (e.g. per-user revenue, click
        indicator, watch-time). For ratio metrics, pass the realised
        per-unit ratio here AND the per-unit ``numerator`` / ``denominator``
        as separate arrays.
    covariate
        Optional 1-D array of pre-experiment covariate values, **paired by
        index** with ``outcomes``. Required for CUPED / CUPAC / paired tests.
    strata
        Optional integer stratum labels (one per row of ``outcomes``).
        Required for ``PostStratification``. If absent and any criterion
        needs strata, pass ``n_strata`` to bin them post-hoc from the
        covariate.
    numerator, denominator
        Optional per-unit numerator and denominator arrays for ratio
        metrics. When both are provided the generator runs in ratio mode:
        bootstrap-resamples ``(N_i, D_i)`` pairs and exposes them as the
        ``numerator_*`` / ``denominator_*`` aux arrays consumed by
        :class:`~absim.criteria.DeltaMethod` and
        :class:`~absim.criteria.Linearization`.
    n_per_group
        Sample size per arm in each Monte Carlo iteration. Independent of
        the size of ``outcomes`` — the bootstrap is *with replacement*, so
        you can simulate experiments much larger than your historical data.
    relative
        If ``True``, the simulator's ``mean_shift`` is interpreted as a
        multiplicative lift (``rate * (1 + shift)``); otherwise as an
        absolute shift on the metric's natural scale.
    n_strata
        If ``strata`` is not provided but a covariate is, bin the covariate
        post-hoc into ``n_strata`` equal-frequency buckets so stratified
        criteria still receive sensible labels. Set to ``1`` to disable.
    name
        Free-form label, propagated into reports.

    Behaviour by metric type
    ------------------------
    The generator auto-detects metric type from the data shape:

    1. **Ratio** — when both ``numerator`` and ``denominator`` are provided.
       Effect injection is on the numerator: ``N_t ← N_t * (1 + mean_shift)``
       in relative mode, or ``N_t ← N_t + mean_shift * D_t`` in absolute mode
       (so the realised per-unit ratio shifts by exactly ``mean_shift``).

    2. **Binary** — when ``outcomes`` contains only the values ``{0, 1}``.
       Effect injection flips a calibrated fraction of zeros to ones (or vice
       versa) in the treatment arm so the empirical mean shifts by exactly
       ``mean_shift`` (absolute) or by ``baseline * mean_shift`` (relative).
       This preserves the binary structure that the z-test for proportions
       expects.

    3. **Continuous** — anything else. Effect injection is a direct additive
       or multiplicative shift on resampled values.
    """

    outcomes: FloatArray
    covariate: FloatArray | None = None
    strata: Any | None = None  # IntArray-like; kept Any to permit any int dtype
    numerator: FloatArray | None = None
    denominator: FloatArray | None = None
    n_per_group: int = 1000
    relative: bool = False
    n_strata: int = 1
    name: str = "empirical"
    _is_binary: bool = field(init=False, repr=False, compare=False, default=False)
    _is_ratio: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(self) -> None:
        out = np.asarray(self.outcomes, dtype=float)
        n = out.size
        if n < 2:
            raise ValueError("EmpiricalGenerator requires at least 2 historical observations")
        object.__setattr__(self, "outcomes", out)

        for label in ("covariate", "numerator", "denominator"):
            arr = getattr(self, label)
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                if arr.size != n:
                    raise ValueError(
                        f"`{label}` must have the same length as `outcomes` (got {arr.size} vs {n})"
                    )
                object.__setattr__(self, label, arr)

        if self.strata is not None:
            strata = np.asarray(self.strata)
            if strata.size != n:
                raise ValueError(
                    f"`strata` must have the same length as `outcomes` (got {strata.size} vs {n})"
                )
            if not np.issubdtype(strata.dtype, np.integer):
                strata = strata.astype(int)
            object.__setattr__(self, "strata", strata)

        ratio_mode = self.numerator is not None and self.denominator is not None
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError(
                "`numerator` and `denominator` must both be provided for ratio mode "
                "(or both omitted)"
            )
        if ratio_mode and float(np.asarray(self.denominator).sum()) == 0.0:
            raise ValueError("denominator sum is zero — cannot bootstrap ratio metric")
        object.__setattr__(self, "_is_ratio", ratio_mode)

        binary_mode = bool(set(np.unique(out).tolist()).issubset(_BINARY_SET))
        object.__setattr__(self, "_is_binary", binary_mode and not ratio_mode)

        if self.n_per_group < 2:
            raise ValueError("n_per_group must be >= 2")
        if self.n_strata < 1:
            raise ValueError("n_strata must be >= 1")

    # --------------------------- internals -------------------------------

    def _inject_continuous(self, sample: FloatArray, mean_shift: float) -> FloatArray:
        """Apply additive or relative shift to a resampled continuous arm."""
        if mean_shift == 0.0:
            return sample
        if self.relative:
            return sample * (1.0 + mean_shift)
        return sample + mean_shift

    def _inject_binary(
        self, sample: FloatArray, mean_shift: float, rng: np.random.Generator
    ) -> FloatArray:
        """Flip a calibrated fraction so empirical mean shifts by ``mean_shift``."""
        if mean_shift == 0.0:
            return sample
        baseline = float(np.mean(self.outcomes))
        if self.relative:
            target = baseline * (1.0 + mean_shift)
        else:
            target = baseline + mean_shift
        target = float(np.clip(target, 0.0, 1.0))
        if target > baseline:
            zeros_mask = sample == 0.0
            denom = 1.0 - baseline
            if denom <= 0.0 or not zeros_mask.any():
                return sample
            flip_p = (target - baseline) / denom
            flip = rng.uniform(size=sample.size) < flip_p
            return np.where(zeros_mask & flip, 1.0, sample)
        ones_mask = sample == 1.0
        if baseline <= 0.0 or not ones_mask.any():
            return sample
        flip_p = (baseline - target) / baseline
        flip = rng.uniform(size=sample.size) < flip_p
        return np.where(ones_mask & flip, 0.0, sample)

    def _inject_ratio_numerator(
        self, num_t: FloatArray, den_t: FloatArray, mean_shift: float
    ) -> FloatArray:
        """Lift the numerator so the realised per-unit ratio shifts by ``mean_shift``."""
        if mean_shift == 0.0:
            return num_t
        if self.relative:
            return num_t * (1.0 + mean_shift)
        return num_t + mean_shift * den_t

    def _build_aux(
        self,
        idx_t: np.ndarray[Any, np.dtype[np.integer[Any]]],
        idx_c: np.ndarray[Any, np.dtype[np.integer[Any]]],
    ) -> dict[str, Any]:
        """Pack the auxiliary arrays expected by every criterion family."""
        aux: dict[str, Any] = {}
        if self.covariate is not None:
            cov = np.asarray(self.covariate)
            cov_t = cov[idx_t]
            cov_c = cov[idx_c]
            aux["covariate_treatment"] = cov_t
            aux["covariate_control"] = cov_c
            aux["features_treatment"] = cov_t.reshape(-1, 1)
            aux["features_control"] = cov_c.reshape(-1, 1)
        if self.strata is not None:
            strata = np.asarray(self.strata)
            aux["strata_treatment"] = strata[idx_t]
            aux["strata_control"] = strata[idx_c]
        elif self.covariate is not None and self.n_strata > 1:
            from absim.generators.base import make_strata

            cov_t = aux["covariate_treatment"]
            cov_c = aux["covariate_control"]
            strata_t, strata_c = make_strata(cov_t, cov_c, self.n_strata)
            aux["strata_treatment"] = strata_t
            aux["strata_control"] = strata_c
        return aux

    # ----------------------------- API -----------------------------------

    def sample(self, rng: np.random.Generator, mean_shift: float) -> Sample:
        """Draw one Monte Carlo bootstrap sample with calibrated effect injection."""
        n = int(np.asarray(self.outcomes).size)
        idx_t = rng.integers(0, n, size=self.n_per_group)
        idx_c = rng.integers(0, n, size=self.n_per_group)

        if self._is_ratio:
            num = np.asarray(self.numerator)
            den = np.asarray(self.denominator)
            num_t = num[idx_t]
            den_t = den[idx_t]
            num_c = num[idx_c]
            den_c = den[idx_c]
            num_t = self._inject_ratio_numerator(num_t, den_t, mean_shift)
            t = num_t / np.maximum(den_t, 1.0)
            c = num_c / np.maximum(den_c, 1.0)
            aux = self._build_aux(idx_t, idx_c)
            aux["numerator_treatment"] = num_t
            aux["numerator_control"] = num_c
            aux["denominator_treatment"] = den_t
            aux["denominator_control"] = den_c
            return Sample(treatment=t, control=c, aux=aux)

        out = np.asarray(self.outcomes)
        c = out[idx_c].astype(float)
        t = out[idx_t].astype(float)
        if self._is_binary:
            t = self._inject_binary(t, mean_shift, rng)
        else:
            t = self._inject_continuous(t, mean_shift)
        aux = self._build_aux(idx_t, idx_c)
        return Sample(treatment=t, control=c, aux=aux)
