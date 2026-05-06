# Architecture

This document explains how `absim` is laid out, why each architectural
decision was made, and where `absim` deviates from naive implementations of
the underlying statistical methods.

## Why this project exists

Most working data scientists pick A/B-test criteria by **rule of thumb**:
"Welch for continuous, z-test for proportions, CUPED if we have a pre-period
metric, bootstrap if I don't trust the parametric assumptions." Sometimes
the rule is wrong — for *your* sample size, *your* data shape, *your* effect
size — and the team only finds out after shipping a miscalibrated test or
chasing a non-result that a more powerful criterion would have caught.

`absim` exists so that question stops being a guess. The simulator drives
off **your real production data** via `EmpiricalGenerator` (bootstrap-
resamples from observed historical outcomes), or off one of three
parametric generators when historical data isn't available. Pick a list of
criteria to compare, and `absim` runs **10 000+ synthetic experiments** and
reports, for each criterion:

- the **false-positive rate** under H₀, with a Wilson 95% CI so you can
  separate noise from real miscalibration;
- the **power** under H₁ for the effect sizes you care about.

The library consolidates the best statistical practice — variance-reduction
methods (CUPED, CUPAC, post-stratification, paired stratification),
ratio-metric criteria (delta-method, Budylin linearization), bootstrap
(percentile + BCa) — under one `Criterion` Protocol so swapping criteria is
a one-line change. The simulator is fast (10 000 Welch sims in ~1.3 s),
parallel, and reproducible bit-for-bit from a single integer seed.

The `EmpiricalGenerator` design is what closes the "but it's all synthetic"
objection: the bootstrap inherits whatever quirks live in your warehouse
data — zero inflation, heavy tails, multimodality, clipping — that the
parametric generators won't reproduce.

## High-level shape

```
                ┌──────────────────────────────────────────────────────┐
                │                    absim.Simulator                   │
                │  ┌────────────────────────────────────────────────┐  │
                │  │ for sim in 1..N (parallel via joblib):         │  │
                │  │     rng     ← seed.spawn()                     │  │
                │  │     sample  ← generator.sample(rng, shift)     │  │
                │  │     result  ← criterion.test(sample, **aux)    │  │
                │  └────────────────────────────────────────────────┘  │
                │                            │                         │
                │       aggregate → SimulationReport (Wilson CI)       │
                └──────────────────────────────────────────────────────┘
                       ▲                                ▲
                       │                                │
            ┌──────────┴──────────┐         ┌───────────┴──────────┐
            │ absim.generators.*  │         │  absim.criteria.*    │
            │ ContinuousGenerator │         │  WelchTTest          │
            │ BinaryGenerator     │         │  CUPED, CUPAC        │
            │ RatioGenerator      │         │  Bootstrap (pct/BCa) │
            └─────────────────────┘         │  DeltaMethod         │
                                            │  Linearization       │
                                            │  PostStratification  │
                                            │  PairedStratification│
                                            │  ZTestProportions    │
                                            └──────────────────────┘
```

Three primitives compose: **generators** produce a `Sample(treatment,
control, aux)`, **criteria** consume one and return a `TestResult`, and the
**simulator** runs the loop, aggregates rejections into FPR/Power with a
Wilson confidence band, and persists results.

## Decision log

| Area | Choice | Why |
|---|---|---|
| Build / dep manager | `uv` | Fast, modern, native lockfile, manages CPython installs from one binary. |
| Build backend | `hatchling` | Lightweight, plays well with `uv`, no plugin sprawl. |
| Lint / format | `ruff` (check + format) | Single binary covers isort, black, pyflakes, pyupgrade, pydocstyle. |
| Type checker | `mypy --strict` on `src/absim`; relaxed in tests | Strict for the library surface; pragmatic for tests where introspection is hairy. |
| Tests | `pytest` + `pytest-cov` + `hypothesis` | Industry standard. Property tests for stat invariants. |
| Parallelism | `joblib.Parallel(backend="loky")` | Process-based, sidesteps the GIL, robust against NumPy state. |
| Plotting | `matplotlib` + `seaborn` | Static, reportable, no JS bundle to ship. |
| Tabular IO | `pandas` + `pyarrow` (parquet) | Columnar, fast, language-agnostic. |
| Config | Hydra **structured configs** (dataclasses + ConfigStore) | Type-checked, override-friendly, plays well with the CLI. |
| CLI | `absim` (argparse top, Hydra inside `run`) | `--help` works without Hydra; full overrides available inside. |
| Docs | `mkdocs` + `mkdocs-material` + `mkdocstrings[python]` | Faster to set up than Sphinx, prettier defaults, autogen API ref. |
| Versioning | SemVer, starting at `0.1.0` | Communicates stability honestly. |
| License | MIT | Most permissive; broadest adoption for an OSS stats tool. |
| Random API | `numpy.random.Generator` only (no legacy `np.random.*`) | Reproducible per-iteration streams via `SeedSequence.spawn()`. |
| Criterion registry | `@register("name")` decorator + `dict` lookup | Adding a criterion = one new module + one decorator line. |

## How a single Monte Carlo iteration is reproducible

1. The user passes an integer `seed` to `Simulator`.
2. The simulator builds a `numpy.random.SeedSequence(seed)` and calls
   `.spawn(n_sims)` to obtain `n_sims` *independent* child SeedSequences.
3. Each child seed becomes a fresh `np.random.default_rng(seed)` inside its
   worker process. Generators and the bootstrap criterion both consume
   that single rng, so any iteration is bit-identical regardless of whether
   the simulator is run serially or across N workers.

## Auxiliary-data convention

Generators emit every auxiliary array that *any* criterion might need
(covariate, strata, numerator/denominator, features for CUPAC) under
well-known keys (`covariate_treatment`, `numerator_control`, etc.).
Criteria pluck the keys they need from `**kwargs`. Unrecognised keys are
silently ignored. This keeps the simulator a one-liner —
`criterion.test(sample.treatment, sample.control, **sample.aux)` — and lets
the registry stay decoupled from the data model.

## Deviations and design notes

### Paired stratification requires paired data

The textbook matched-pair test needs *truly paired* observations — the i-th
treatment unit and the i-th control unit must share a within-pair latent
covariate, otherwise the within-pair differences are positively
autocorrelated through the order-statistic structure of independent
samples and the t-statistic over-rejects. We surface this honestly:

- `PairedStratification(paired=True)` (default) takes
  `T_i - C_i` directly, which is the correct paired test.
- `ContinuousGenerator(paired=True)` produces genuinely paired samples
  (T and C share a within-pair covariate).
- `paired=False` falls back to rank-matching for completeness, but is
  documented as having a small FPR inflation. We do not include it in
  experiments where `paired=False` is the data-generating process.

### CUPAC is fit pooled, not per-arm

A naive CUPAC implementation does out-of-fold predictions *within each
arm*. Under H₁ this leaks the treatment effect into the predictor, which
then absorbs the very signal we wanted to test for, destroying power.
Our `_oof_predict_pooled` fits a single regression of `Y` on `X` over the
**pooled** sample. Predictions are symmetric in arm assignment, the
average treatment effect is preserved, and only the within-arm variance is
reduced — exactly what the CUPED transform needs.

### Bootstrap p-value

The percentile-bootstrap p-value here is `2 * min(P(diff ≤ 0),
P(diff ≥ 0))` — the achieved-significance-level formulation. The CI-based
rejection rule (`ci excludes 0`) is the primary decision rule; the p-value
is reported for completeness and is consistent with that rule for
percentile intervals.

### Delta-method vs linearization

Asymptotically equivalent. We ship both because:

- Delta-method gives a closed-form variance and a clean z-test.
- Linearization produces a per-unit additive metric `L_i`, which lets you
  drop in **any** further variance reduction (CUPED, post-stratification)
  on top of the linearised values. This is the Budylin recipe.

### Wilson interval for the rejection-rate band

We use the Wilson score interval rather than the normal approximation.
Around the typical α = 0.05 with N = 10 000 sims, the difference is
small, but Wilson is markedly more accurate for small-N runs and avoids
intervals slipping outside [0, 1].

## Where to read code first

1. `src/absim/types.py` — `TestResult`, `EffectSize`, `SimulationReport`.
2. `src/absim/criteria/base.py` — the `Criterion` Protocol and registry.
3. `src/absim/simulator.py` — the loop and seed splitting.
4. `src/absim/criteria/welch.py` — the simplest concrete criterion;
   everything else is variations on this theme.
5. `src/absim/generators/continuous.py` — the canonical generator and its
   auxiliary-data emission scheme.
