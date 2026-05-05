# Tutorial: from synthetic data to a calibration report

This walkthrough uses Welch's t-test on a Gaussian outcome to demonstrate
the full `absim` workflow. Once you've finished it, swapping in any other
criterion or generator is a one-line change.

## 1. Synthesise data

Pick a generator. The continuous generator emits a Gaussian outcome plus a
pre-experiment covariate (so CUPED has something to work with) and strata
indicators (for `PostStratification`).

```python
from absim.generators import ContinuousGenerator

gen = ContinuousGenerator(
    n_per_group=1000,    # users per arm
    mean=0.0,            # baseline mean
    sd=1.0,              # outcome noise
    rho=0.6,             # corr(Y, covariate)
    n_strata=4,          # equal-probability strata to emit
)
```

Drawing one sample by hand:

```python
import numpy as np

rng = np.random.default_rng(0)
sample = gen.sample(rng, mean_shift=0.1)  # +0.1 lift on treatment
sample.treatment.shape, sample.aux.keys()
# ((1000,), dict_keys(['covariate_treatment', 'covariate_control', ...]))
```

## 2. Pick a criterion

Every criterion is a small dataclass:

```python
from absim.criteria import CUPED, WelchTTest

welch = WelchTTest(alpha=0.05)
cuped = CUPED(alpha=0.05)
```

Single-shot test on `sample`:

```python
result = welch.test(sample.treatment, sample.control, **sample.aux)
result.p_value, result.effect, result.ci_low, result.ci_high
```

## 3. Drive a Monte Carlo simulation

```python
from absim import EffectSize, Simulator

sim_h0 = Simulator(
    generator=gen,
    criterion=welch,
    n_sims=5000,
    alpha=0.05,
    effect=EffectSize(name="none", value=0.0),  # H0 → FPR
    seed=0,
)
report_h0 = sim_h0.run()                 # parallel by default
print(report_h0.fpr, report_h0.binomial_ci_low, report_h0.binomial_ci_high)
```

For power, swap the effect size:

```python
sim_h1 = Simulator(
    generator=gen,
    criterion=welch,
    n_sims=5000,
    alpha=0.05,
    effect=EffectSize(name="medium", value=0.1),
    seed=0,
)
print(sim_h1.run().power)
```

## 4. Compare criteria, plot the result

```python
from absim.reports import plot_power_curve

reports = []
for crit in (welch, cuped):
    for value in (0.0, 0.05, 0.1, 0.2):
        sim = Simulator(
            generator=gen, criterion=crit, n_sims=2000, alpha=0.05,
            effect=EffectSize(name=str(value), value=value), seed=0,
        )
        reports.append(sim.run())

fig = plot_power_curve(reports)
fig.savefig("power.png", dpi=120)
```

## 5. Or use the CLI

Everything above is wired to a Hydra config — running an experiment is a
single command:

```bash
absim run experiment=continuous_welch_vs_cuped simulator.n_sims=10000
```

It writes `outputs/.../reports.parquet`, `reports.csv`, `fpr.png`, and
`power.png`. Add `--help` after `run` to see all override flags.
