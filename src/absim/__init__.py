"""absim — Monte Carlo validator for A/B testing criteria.

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
