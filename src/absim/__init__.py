"""absim — pick the right A/B-test criterion before you ship it to prod.

`absim` runs 10 000+ synthetic A/B experiments and reports, for every
criterion you compare:

- **calibration** — is the false-positive rate really α?
  (with a Wilson confidence band so you can tell noise from miscalibration)
- **power** — for the effect size you actually care about, on the sample
  size you actually have, on the data shape you actually see.

Use it to: choose between ``WelchTTest``, ``Bootstrap``, ``DeltaMethod``,
``Linearization`` and friends; quantify the variance-reduction win from
``CUPED``, ``CUPAC``, ``PostStratification``; or sanity-check your in-house
test code by running it on synthetic data with a known true effect.

The package exposes:

- :mod:`absim.criteria` — statistical criteria (Welch, CUPED, bootstrap, ...).
- :mod:`absim.generators` — synthetic data generators (continuous, binary, ratio).
- :mod:`absim.simulator` — the Monte Carlo engine.
- :mod:`absim.reports` — aggregation & visualisation.

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
