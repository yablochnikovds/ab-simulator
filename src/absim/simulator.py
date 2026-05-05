"""Monte Carlo simulator engine.

The simulator drives ``n_sims`` independent (data, criterion) iterations.
Each iteration draws a fresh :class:`~absim.generators.base.Sample` and runs
``criterion.test(...)``. The aggregated rejection rate estimates **FPR**
under H₀ (``effect.value == 0``) and **Power** under H₁ otherwise.

Reproducibility
---------------
A single integer ``seed`` is split via :class:`numpy.random.SeedSequence`
into ``n_sims`` independent child seeds. Each iteration receives one — so
any iteration is bit-identical regardless of parallelism.

Parallelism
-----------
Uses :class:`joblib.Parallel` with the ``"loky"`` (process) backend by
default. Process-based parallelism sidesteps the GIL and is robust against
NumPy/SciPy state. Set ``parallel=False`` for serial mode (useful in tests
and deterministic benchmarking).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from joblib import Parallel, delayed

from absim._stats import wilson_ci
from absim.types import EffectSize, SimulationReport

if TYPE_CHECKING:
    from absim.criteria.base import Criterion
    from absim.generators.base import Generator
    from absim.types import TestResult


def _run_one(
    generator: Generator,
    criterion: Criterion,
    seed: int | np.random.SeedSequence,
    mean_shift: float,
) -> TestResult:
    """Run a single simulation iteration."""
    rng = np.random.default_rng(seed)
    sample = generator.sample(rng, mean_shift)
    return criterion.test(sample.treatment, sample.control, rng=rng, **sample.aux)


@dataclass(slots=True)
class Simulator:
    """Monte Carlo simulator for one (generator, criterion) pair.

    Parameters
    ----------
    generator
        The synthetic data generator.
    criterion
        The criterion to evaluate.
    n_sims
        Number of Monte Carlo iterations.
    alpha
        Significance level (used both as the criterion's rejection threshold
        and as a sanity cross-check in reports).
    effect
        The effect size to inject. ``EffectSize(name="none", value=0.0)``
        ⇒ FPR estimation. Non-zero ⇒ Power estimation.
    seed
        Root seed; split deterministically into per-iteration seeds.
    n_jobs
        Number of parallel workers when ``parallel=True``. ``-1`` ⇒ all cores.
    """

    generator: Generator
    criterion: Criterion
    n_sims: int = 10_000
    alpha: float = 0.05
    effect: EffectSize = field(default_factory=lambda: EffectSize(name="none", value=0.0))
    seed: int = 0
    n_jobs: int = -1

    def run(self, *, parallel: bool = True) -> SimulationReport:
        """Execute the simulation and return an aggregated :class:`SimulationReport`."""
        if self.n_sims <= 0:
            raise ValueError("n_sims must be positive")
        ss = np.random.SeedSequence(self.seed)
        child_seeds = ss.spawn(self.n_sims)
        start = time.perf_counter()
        results = self._dispatch(child_seeds, parallel=parallel)
        runtime = time.perf_counter() - start

        rejections = sum(1 for r in results if r.rejected)
        rate = rejections / self.n_sims
        ci_low, ci_high = wilson_ci(rejections, self.n_sims, confidence=0.95)
        p_values = np.fromiter((r.p_value for r in results), dtype=float, count=self.n_sims)
        effects = np.fromiter((r.effect for r in results), dtype=float, count=self.n_sims)
        ses = np.fromiter((r.std_error for r in results), dtype=float, count=self.n_sims)
        mean_se = float(np.nanmean(ses))
        return SimulationReport(
            criterion_name=self.criterion.name,
            n_sims=self.n_sims,
            alpha=self.alpha,
            effect=self.effect,
            rejection_rate=rate,
            binomial_ci_low=ci_low,
            binomial_ci_high=ci_high,
            mean_pvalue=float(np.mean(p_values)),
            mean_effect=float(np.mean(effects)),
            mean_std_error=mean_se,
            runtime_sec=runtime,
            metadata={
                "generator": getattr(self.generator, "name", type(self.generator).__name__),
                "n_jobs": self.n_jobs if parallel else 1,
            },
        )

    def _dispatch(
        self,
        child_seeds: list[np.random.SeedSequence],
        *,
        parallel: bool,
    ) -> list[TestResult]:
        """Run all iterations either serially or via ``joblib``."""
        if parallel and self.n_jobs not in (0, 1):
            parallel_runner: Any = Parallel(n_jobs=self.n_jobs, backend="loky")
            tasks = (
                delayed(_run_one)(self.generator, self.criterion, s, self.effect.value)
                for s in child_seeds
            )
            return list(parallel_runner(tasks))
        return [_run_one(self.generator, self.criterion, s, self.effect.value) for s in child_seeds]
