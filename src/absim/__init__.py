"""absim — power-analyze A/B-test criteria on your real production data.

`absim` runs 10 000+ synthetic A/B experiments and reports, for every
criterion you compare:

- **calibration** — is the false-positive rate really α?
  (with a Wilson confidence band so you can tell noise from miscalibration)
- **power** — for the effect size you actually care about, on the sample
  size you actually have, on the data shape you actually see.

The synthetic data is bootstrap-resampled from arrays *you* hand to
:class:`absim.generators.EmpiricalGenerator` — so the simulation reflects
the actual distribution of your production metric, including its quirks
(zero-inflation, heavy tails, multi-modality, clipping artifacts) that
parametric generators won't reproduce. Three parametric generators
(:class:`~absim.generators.ContinuousGenerator`,
:class:`~absim.generators.BinaryGenerator`,
:class:`~absim.generators.RatioGenerator`) are also provided for cases when
historical data isn't available.

Use it to: choose between ``WelchTTest``, ``Bootstrap``, ``DeltaMethod``,
``Linearization`` and friends; quantify the variance-reduction win from
``CUPED``, ``CUPAC``, ``PostStratification``; sanity-check your in-house
test code via a calibration audit on real data; or run a methodology RFC
with reproducible artifacts.

The package exposes:

- :mod:`absim.criteria` — statistical criteria (Welch, CUPED, bootstrap, ...).
- :mod:`absim.generators` — synthetic data generators (continuous, binary, ratio).
- :mod:`absim.simulator` — the Monte Carlo engine.
- :mod:`absim.reports` — aggregation & visualization.

Example
-------
>>> from absim import EffectSize, Simulator
>>> from absim.criteria import WelchTTest
>>> from absim.generators import ContinuousGenerator
>>> gen = ContinuousGenerator(n_per_group=500, rho=0.0)
>>> sim = Simulator(
...     generator=gen,
...     criterion=WelchTTest(),
...     n_sims=200,
...     effect=EffectSize(name="small", value=0.2),
...     seed=0,
... )
>>> report = sim.run(parallel=False)
>>> isinstance(report.power, float)
True
"""

from __future__ import annotations

from absim._version import __version__
from absim.simulator import Simulator
from absim.types import EffectSize, SimulationReport, TestResult

__all__ = [
    "EffectSize",
    "SimulationReport",
    "Simulator",
    "TestResult",
    "__version__",
]
