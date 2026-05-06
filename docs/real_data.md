# Power analysis on your real data

The textbook objection to a Monte Carlo simulator is "but it generates fake
data — my real metric doesn't look like a Gaussian". `EmpiricalGenerator`
is the answer: it bootstrap-resamples from **historical observations you
hand it** and injects a calibrated effect into the treatment arm. The
synthetic experiments inherit your data's quirks — zero-inflation, heavy
tails, multimodality, clipping artifacts — that the parametric generators
won't reproduce.

This page walks through three real-world workflows.

## Workflow 1 — power for a continuous metric (revenue, time-on-page)

You're about to run an A/B test on per-user revenue. You pulled a
representative month from the warehouse. You want to know: at the lift you
hope for, how often will the test reject H₀?

```python
import numpy as np
import pandas as pd
from absim import EffectSize, Simulator
from absim.criteria import Bootstrap, WelchTTest
from absim.generators import EmpiricalGenerator

# Replace with your real warehouse pull.
df = pd.read_parquet("revenue_last_month.parquet")
revenue = df["revenue"].to_numpy(dtype=float)        # zero-inflated, heavy tail
print(f"n={revenue.size}, mean={revenue.mean():.2f}, "
      f"median={np.median(revenue):.2f}, max={revenue.max():.0f}")

gen = EmpiricalGenerator(outcomes=revenue, n_per_group=10_000, relative=True)

for crit in (WelchTTest(), Bootstrap(method="bca", n_resamples=1000)):
    for lift in (0.0, 0.02, 0.05):
        sim = Simulator(gen, crit, n_sims=2000,
                        effect=EffectSize(f"+{lift:.0%}", lift), seed=0)
        r = sim.run()
        label = "FPR" if lift == 0.0 else "Power"
        print(f"  {crit.name:>10s}  lift={lift:+.0%}  {label}={r.rejection_rate:.3f}")
```

Read the FPR column to confirm calibration; read the Power columns at the
lift you actually expect to ship.

**Why bootstrap here?** With heavy-tailed revenue, the `WelchTTest` can
over-reject (FPR drifts above 0.05) even at moderate sample sizes.
`Bootstrap(method="bca")` stays calibrated.

## Workflow 2 — sanity-check your in-house statistical code

Your prod platform has its own `experiments.welch_test()` (or its own
bootstrap, or its own CUPED implementation). Bugs in here are catastrophic
because they show up across hundreds of experiments. You want a calibration
audit.

```python
from dataclasses import dataclass
from absim import Simulator
from absim.criteria.base import register
from absim.types import TestResult
from absim.generators import EmpiricalGenerator

# Your in-house module:
import experiments_platform as platform


@register("inhouse")
@dataclass(frozen=True, slots=True)
class InHouseTest:
    alpha: float = 0.05
    name: str = "inhouse"

    def test(self, treatment, control, **aux) -> TestResult:
        result = platform.welch_test(treatment, control, alpha=self.alpha)
        return TestResult(
            p_value=result.p_value,
            statistic=result.statistic,
            effect=result.point_estimate,
            std_error=result.std_error,
            ci_low=result.ci[0],
            ci_high=result.ci[1],
            rejected=result.p_value < self.alpha,
        )


# Use real data so the audit is realistic, not toy.
gen = EmpiricalGenerator(outcomes=revenue, n_per_group=5000)
report = Simulator(gen, InHouseTest(), n_sims=10_000, seed=0).run()

print(f"In-house FPR: {report.fpr:.4f} "
      f"(95% Wilson CI: [{report.binomial_ci_low:.4f}, "
      f"{report.binomial_ci_high:.4f}])")
# If 0.05 ∉ CI, your in-house code is mis-calibrated and you have a bug.
```

If the Wilson CI brackets α (0.05), the in-house code is calibrated. If
not, the simulation just caught a real production bug.

## Workflow 3 — variance reduction on your CTR metric

You have a ratio metric (CTR per session) and a pre-experiment per-user
covariate (last-week activity). Should you adopt CUPED? Compare on real
historical data.

```python
from absim.criteria import CUPED, WelchTTest
from absim.generators import EmpiricalGenerator

# Real warehouse pull: per-user click rate, paired with last-week sessions.
df = pd.read_parquet("user_metrics_last_month.parquet")

gen = EmpiricalGenerator(
    outcomes=df["click_rate"].to_numpy(dtype=float),
    covariate=df["sessions_last_week"].to_numpy(dtype=float),
    n_per_group=5000,
)

for crit in (WelchTTest(), CUPED()):
    sim = Simulator(gen, crit, n_sims=5000,
                    effect=EffectSize("+5pp", 0.05), seed=0)
    r = sim.run()
    print(f"{crit.name:>8s}  power = {r.power:.3f}")
```

If the CUPED line beats Welch by a meaningful margin (e.g. +20 percentage
points of power), the pre-period covariate is informative enough to be
worth the engineering work.

## When `EmpiricalGenerator` is the right choice — and when it isn't

✅ Use it when you have **an actual sample** of historical outcomes that
captures the distributional structure of your future experiments. The
bootstrap respects whatever weirdness is in the sample (point masses at 0,
heavy tails, gaps, modes).

⚠️ The bootstrap **cannot extrapolate**: if your historical sample doesn't
contain a regime your live experiment will see (e.g. a Black Friday spike),
the simulation will under-represent it. Pull a sample that covers the
regime you'll experiment under.

⚠️ For very small historical samples (< ~500 units), the bootstrap is
high-variance. Either pull more data or fall back to a parametric
`ContinuousGenerator` / `BinaryGenerator` / `RatioGenerator` with parameters
estimated from the small sample.

❌ Don't use it for **forward-looking sample-size formulas**. If you just
need "how many users for 80% power at MDE 5%", `statsmodels.stats.power`
or G\*Power is the right tool — they're a closed-form calculation, not a
Monte Carlo run.

## Reproducibility

`EmpiricalGenerator` is bit-identical between `Simulator(parallel=False)`
and `parallel=True` runs given the same seed. The bootstrap indices are
drawn from the per-iteration RNG produced by `SeedSequence.spawn()`, so
re-running the same simulation with the same seed always returns the same
report — important when these reports become artefacts attached to a PR or
methodology RFC.
