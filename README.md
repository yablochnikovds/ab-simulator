# absim

[![CI](https://github.com/yablochnikovds/ab-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/yablochnikovds/ab-simulator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yablochnikovds/ab-simulator/branch/main/graph/badge.svg)](https://codecov.io/gh/yablochnikovds/ab-simulator)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://pypi.org/project/absim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Power-analyze A/B-test criteria on your real production data — not on toy Gaussians.**

You're about to ship an experiment. Should you use a t-test or a bootstrap?
Will CUPED actually buy you 30% more power, or is the team being optimistic?
Is your in-house variance-reduction code even calibrated?

`absim` is a **Monte Carlo simulator for A/B-test statistical criteria**.
Hand it a NumPy array of historical outcomes from your warehouse — it
bootstrap-resamples your real distribution, injects a calibrated effect into
the treatment arm, runs 10 000+ synthetic experiments, and reports the
**false-positive rate** (with a Wilson confidence band) and the **power**
for each criterion. The synthetic experiments inherit your data's quirks
(zero-inflation, heavy tails, multi-modality) — the things parametric
generators won't reproduce. No more guessing whether "the textbook formula
applies to *our* metric".

```text
        Generator                  Criterion                Simulator
   ┌─────────────────┐         ┌──────────────┐         ┌──────────────┐
   │ Continuous /    │   ─→    │ Welch        │   ─→    │  10 000 runs │
   │ Binary / Ratio  │         │ CUPED        │         │  parallelism │
   │ + covariates    │         │ Bootstrap    │         │  reproducible│
   │ + strata        │         │ Delta-method │         │              │
   │ + paired arms   │         │ + 5 more     │         └──────┬───────┘
   └─────────────────┘         └──────────────┘                ↓
                                                       ┌──────────────┐
                                                       │   Report     │
                                                       │ FPR ± CI     │
                                                       │ Power curves │
                                                       │ Parquet+CSV  │
                                                       └──────────────┘
```

---

## Who it's for

Real situations where `absim` saves you from shipping the wrong test:

### 1. "Power-analyze a real metric from our warehouse"

Your team is about to A/B test a feature on per-user revenue. The metric is
zero-inflated and heavy-tailed; the textbook t-test power formula assumes
neither. Pull a historical sample, hand it to `EmpiricalGenerator`, and let
absim simulate the experiment thousands of times on **your actual
distribution**:

```python
import numpy as np, pandas as pd
from absim import EffectSize, Simulator
from absim.criteria import Bootstrap, WelchTTest
from absim.generators import EmpiricalGenerator

# Pull real historical revenue (zero-inflated, heavy tail).
revenue = pd.read_parquet("revenue_last_month.parquet")["revenue"].to_numpy(float)

gen = EmpiricalGenerator(outcomes=revenue, n_per_group=10_000, relative=True)

for crit in (WelchTTest(), Bootstrap(method="bca", n_resamples=1000)):
    for lift in (0.0, 0.02, 0.05):
        r = Simulator(gen, crit, n_sims=2000,
                      effect=EffectSize(f"+{lift:.0%}", lift), seed=0).run()
        kind = "FPR" if lift == 0 else "Power"
        print(f"{crit.name:>10s}  lift={lift:+.0%}  {kind}={r.rejection_rate:.3f}")
```

Read the FPR rows to confirm the criterion is calibrated **on your data**;
read the Power rows to decide which one is sensitive enough at the lift you
expect to ship.

See [docs/real_data.md](docs/real_data.md) for the full real-data workflow,
including in-house code validation and CTR-with-CUPED on warehouse pulls.

### 2. "Will CUPED actually be worth the effort?"

Your team has a pre-experiment metric strongly correlated with the outcome
(`ρ ≈ 0.8`). Theory says CUPED reduces residual variance by `1 − ρ² = 0.36`
— but the team wants a number, not a formula. Run it:

```python
from absim import EffectSize, Simulator
from absim.criteria import CUPED, WelchTTest
from absim.generators import ContinuousGenerator

gen = ContinuousGenerator(n_per_group=5000, rho=0.8)
for crit in (WelchTTest(), CUPED()):
    sim = Simulator(generator=gen, criterion=crit, n_sims=10_000,
                    effect=EffectSize("small", 0.05), seed=0)
    r = sim.run()
    print(f"{crit.name:>8s}  power = {r.power:.3f}")
# welch_t   power = 0.699   ← finds 70% of real lifts
#    cuped  power = 0.987   ← finds 99% — variance reduction is real
```

### 3. "Is my in-house t-test correctly calibrated?"

Your team rolled a custom Welch / bootstrap implementation in a production
experiment platform. You want a sanity check on **realistic data**: under
H₀ (no real effect), does it reject 5% of the time?

```python
from dataclasses import dataclass
from absim import Simulator
from absim.criteria.base import register
from absim.types import TestResult
from absim.generators import EmpiricalGenerator
import experiments_platform as platform   # your in-house module

@register("inhouse")
@dataclass(frozen=True, slots=True)
class InHouseTest:
    alpha: float = 0.05
    name: str = "inhouse"
    def test(self, treatment, control, **aux) -> TestResult:
        r = platform.welch_test(treatment, control, alpha=self.alpha)
        return TestResult(p_value=r.p_value, statistic=r.statistic,
                          effect=r.point_estimate, std_error=r.std_error,
                          ci_low=r.ci[0], ci_high=r.ci[1],
                          rejected=r.p_value < self.alpha)

# Run the audit on REAL data, not on a Gaussian toy. `revenue` is the
# warehouse pull from Example #1 above.
gen = EmpiricalGenerator(outcomes=revenue, n_per_group=5000)
report = Simulator(gen, InHouseTest(), n_sims=10_000, seed=0).run()
print(f"FPR={report.fpr:.4f}  Wilson 95% CI=[{report.binomial_ci_low:.4f}, "
      f"{report.binomial_ci_high:.4f}]")
# If 0.05 ∉ CI, your in-house code is miscalibrated → bug to fix.
```

### 4. "Delta-method, linearization, or bootstrap for my CTR?"

Ratio metrics (clicks/sessions, ARPU, conversion-rate-per-user) are notorious
because numerator and denominator have different granularity. Three
canonical options — which one is honest *and* most powerful on realistic
data?

```python
from absim import EffectSize, Simulator
from absim.criteria import Bootstrap, DeltaMethod, Linearization
from absim.generators import RatioGenerator

gen = RatioGenerator(n_per_group=2000, base_rate=0.2, sessions_mean=5.0)
for crit in (DeltaMethod(), Linearization(), Bootstrap()):
    fpr = Simulator(gen, crit, n_sims=10_000,
                    effect=EffectSize("none", 0.0), seed=0).run().fpr
    print(f"{crit.name:>16s}  FPR = {fpr:.4f}")
```

### 5. "My revenue metric is heavy-tailed — is t-test still safe?"

You've got revenue per user (lognormal-ish). The textbook says "t-test is
robust" but you're not sure for *your* skewness. Compare on a realistic
mixture distribution and check FPR before relying on it.

```python
gen = ContinuousGenerator(n_per_group=1000, distribution="lognormal", sd=1.5)
for crit in (WelchTTest(), Bootstrap(method="bca")):
    fpr = Simulator(gen, crit, n_sims=10_000, seed=0).run().fpr
    print(f"{crit.name:>10s}  FPR = {fpr:.4f}")
```

### 6. "Does post-stratification actually buy me power?"

You're stratifying by platform / device / cohort. Theory says variance can
only go down. But by how much *on your data*?

```python
from absim.criteria import PostStratification
gen = ContinuousGenerator(n_per_group=3000, rho=0.6, n_strata=4)
sim = Simulator(gen, PostStratification(), n_sims=10_000,
                effect=EffectSize("small", 0.05), seed=0)
print(sim.run().power)   # compare against WelchTTest() on the same data
```

---

## Why not just `scipy.stats` + a notebook?

| Capability                                                      | absim | `scipy`/`statsmodels` | `cluster_experiments` | DIY notebook |
|-----------------------------------------------------------------|:-----:|:---------------------:|:---------------------:|:------------:|
| Welch / z-test / paired t-test                                  |  ✅   |          ✅           |          ✅           |      ✅       |
| Empirical bootstrap-from-real-data + calibrated effect injection |  ✅  |          ❌           |        partial        |    custom    |
| CUPED variance reduction                                        |  ✅   |          ❌           |          ✅           |    custom    |
| **CUPAC** (out-of-fold ML predictor as covariate)               |  ✅   |          ❌           |          ❌           |     rare     |
| Post-stratification & matched pairs                             |  ✅   |          ❌           |        partial        |    custom    |
| **BCa bootstrap** (jackknife-accelerated), vectorized           |  ✅   |        partial        |          ❌           |     slow     |
| **Delta-method & Budylin linearization** for ratio metrics      |  ✅   |          ❌           |          ❌           |     rare     |
| Calibration audit: Wilson CI on FPR for any in-house criterion  |  ✅   |          ❌           |        partial        |     rare     |
| One unified `Criterion` Protocol — drop in your own             |  ✅   |          ❌           |          ❌           |     N/A      |
| 10k-sim Monte Carlo engine (parallel, bit-identical reproducible) | ✅ |         N/A           |          ✅           |  hand-rolled |
| Hydra configs + CLI for running experiment grids                |  ✅   |          ❌           |          ❌           |     N/A      |

`cluster_experiments` is the closest sibling — it shines for clustered /
switchback designs and accepts your raw DataFrame for power analysis.
`absim` complements it with **CUPAC, BCa bootstrap, ratio-metric
linearization, and a calibration-audit harness** for vetting in-house
statistical code on real warehouse data.

---

## Install

```bash
# Install from source (PyPI publication is on the way):
git clone https://github.com/yablochnikovds/ab-simulator
cd ab-simulator
uv sync                       # or:  pip install -e .
```

Python 3.10+ required. CI runs on 3.10 / 3.11 / 3.12.

## Quick start

```python
from absim import EffectSize, Simulator
from absim.criteria import CUPED, WelchTTest
from absim.generators import ContinuousGenerator

gen = ContinuousGenerator(n_per_group=1000, sd=1.0, rho=0.6)
for crit in (WelchTTest(), CUPED()):
    sim = Simulator(generator=gen, criterion=crit, n_sims=5_000,
                    effect=EffectSize("medium", 0.1), seed=0)
    r = sim.run()
    print(f"{crit.name:>8s}  power = {r.power:.3f}  "
          f"({r.binomial_ci_low:.3f}, {r.binomial_ci_high:.3f})")
```

```text
 welch_t  power = 0.609  (0.595, 0.622)
   cuped  power = 0.801  (0.789, 0.811)
```

10 000 simulations of Welch's t-test complete in **~1.3 s** on a single M-series
core (parallel, reproducible from a single integer seed).

## Command line

`absim` ships a Hydra-driven CLI for predefined experiments — useful for grid
sweeps and CI artifacts:

```bash
absim list-criteria
absim list-experiments
absim run experiment=continuous_welch_vs_cuped
absim run experiment=ratio_delta_vs_linearization \
    data.n_per_group=2000 simulator.n_sims=20_000
```

Each run drops a parquet/CSV of reports plus `fpr.png` and `power.png` under
`outputs/`.

## What's in the box

**Criteria** (all under one `Criterion` Protocol):

| Family            | Criteria                                                                   |
|-------------------|----------------------------------------------------------------------------|
| Continuous, mean  | `WelchTTest`, `CUPED`, `CUPAC`, `PostStratification`, `PairedStratification` |
| Distribution-free | `Bootstrap` (percentile + BCa)                                             |
| Binary metrics    | `ZTestProportions`                                                         |
| Ratio metrics     | `DeltaMethod`, `Linearization` (Budylin)                                   |

**Generators** with realistic structure (not toy Gaussian):

- **Continuous** — Gaussian / lognormal / mixture; optional pre-experiment
  covariate; optional paired sampling for matched-pair designs.
- **Binary** — Bernoulli outcomes with logistic-link covariate (so CUPED
  has something real to reduce variance against).
- **Ratio** — Poisson sessions × per-user rate, producing genuine
  numerator–denominator correlation; relative or absolute lift.

Each generator emits all auxiliary arrays the criteria need
(`covariate_*`, `strata_*`, `numerator_*`, `denominator_*`,
`features_*` for CUPAC).

## When you might *not* need absim

- If you're running a one-off back-of-the-envelope power calculation, a
  G\*Power or `statsmodels.stats.power` formula is faster.
- If you only ever use a vanilla Welch t-test and trust scipy — you don't
  need a full simulator. Add `absim` to your kit when you start considering
  variance-reduction methods or non-parametric tests.

## Documentation

- 📖 [Tutorial](docs/tutorial.md) — synthesize data → run the simulator → read
  the report.
- 📐 [Criterion reference](docs/criteria/) — formula, intuition, assumptions for
  each criterion (Welch, CUPED, CUPAC, bootstrap, delta-method, …).
- 📊 [Benchmark](docs/benchmark.md) — head-to-head FPR & power across all
  criteria, metric types, and effect sizes.
- 🏗 [Architecture](ARCHITECTURE.md) — design rationale and decision log.
- 🛠 [Contributing](CONTRIBUTING.md) — how to add a criterion or a generator
  in a single file.
- 📓 [Example notebook](examples/cuped_vs_ttest.ipynb) — *CUPED vs t-test:
  variance reduction in practice*.

## License

MIT — see [LICENSE](LICENSE).
