# absim

[![CI](https://github.com/yablochnikovds/ab-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/yablochnikovds/ab-simulator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yablochnikovds/ab-simulator/branch/main/graph/badge.svg)](https://codecov.io/gh/yablochnikovds/ab-simulator)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://pypi.org/project/absim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**absim** is a Monte Carlo validator for the statistical criteria you use in
A/B testing. Pick a criterion (Welch, CUPED, bootstrap, delta-method, ...),
pick a synthetic data generator (continuous / binary / ratio), and absim
will tell you — with binomial confidence bands — whether the criterion is
**well-calibrated** (FPR ≈ α under H₀) and **how powerful** it is for the
effect sizes you care about. Use it to choose criteria, validate
implementations, and quantify variance reduction before committing to them
in production.

## Install

```bash
pip install absim          # not yet on PyPI; install from source for now
# or, with uv:
uv add absim
```

To work on a checkout:

```bash
git clone https://github.com/yablochnikovds/ab-simulator
cd ab-simulator
uv sync
```

## Quick start

```python
from absim import EffectSize, Simulator
from absim.criteria import CUPED, WelchTTest
from absim.generators import ContinuousGenerator

gen = ContinuousGenerator(n_per_group=1000, mean=0.0, sd=1.0, rho=0.6)

for crit in (WelchTTest(), CUPED()):
    sim = Simulator(
        generator=gen,
        criterion=crit,
        n_sims=5000,
        alpha=0.05,
        effect=EffectSize(name="medium", value=0.1),  # H1 ⇒ Power
        seed=0,
    )
    report = sim.run()
    print(f"{crit.name:>8s} Power = {report.power:.3f} "
          f"({report.binomial_ci_low:.3f}, {report.binomial_ci_high:.3f})")
```

```text
 welch_t Power = 0.567 (0.553, 0.581)
   cuped Power = 0.929 (0.921, 0.937)
```

CUPED roughly doubles power on this scenario — exactly what you'd expect from
its variance-reduction theory `1 - ρ² ≈ 0.64`.

## Command line

`absim` ships a Hydra-driven CLI for predefined experiments:

```bash
absim list-criteria
absim list-experiments
absim run experiment=continuous_welch_vs_cuped
absim run experiment=ratio_delta_vs_linearization data.n_per_group=2000 simulator.n_sims=20000
```

Each run drops a parquet/CSV of reports plus `fpr.png` and `power.png` under
`outputs/`.

## What's in the box

| Family            | Criterion                                           |
|-------------------|-----------------------------------------------------|
| Continuous, mean  | `WelchTTest`, `CUPED`, `CUPAC`, `PostStratification`, `PairedStratification` |
| Continuous, all   | `Bootstrap` (percentile + BCa)                      |
| Binary, prop.     | `ZTestProportions`                                  |
| Ratio metrics     | `DeltaMethod`, `Linearization` (Budylin)            |

Generators: continuous (Gaussian / lognormal / mixture, optional pre-experiment
covariate, optional paired sampling), binary (Bernoulli, logistic-link
covariate), ratio (Poisson sessions × per-user rate, realistic N–D
correlation). Each emits all auxiliary arrays the criteria need
(`covariate_*`, `strata_*`, `numerator_*`, etc.).

## Documentation

- [Architecture](ARCHITECTURE.md) — design + decision log.
- [Criterion reference](docs/criteria/) — formula, intuition, assumptions.
- [Benchmark](docs/benchmark.md) — power & FPR table across all criteria.
- [Contributing](CONTRIBUTING.md) — how to add a criterion or a generator.

## License

MIT — see [LICENSE](LICENSE).
