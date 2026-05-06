# Changelog

All notable changes to `absim` will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`absim.generators.EmpiricalGenerator`** — bootstrap-resamples from real
  observed data (NumPy arrays / pandas columns from your warehouse) and
  injects calibrated effects into the treatment arm. Auto-detects
  continuous / binary / ratio metric types and supports paired covariates,
  strata, and numerator/denominator pairs. Closes the "but I can't run
  absim on my real data" objection: the simulator now drives off your
  actual distribution.
- `_stats.make_result(...)` and `_stats.degenerate_result(...)` helpers and a
  `_stats.two_sided_t_pvalue(...)` p-value helper that absorb the
  `TestResult` boilerplate from every criterion.
- `generators.base.make_strata(...)` shared helper used by all three
  parametric generators for equal-frequency stratum binning.

### Changed

- **`BinaryGenerator`** now numerically calibrates the logistic slope
  ``β`` (via Brent's method) so the realised ``corr(Y, X)`` actually
  matches the requested ``rho`` to ≈ 1e-3, instead of the previous loose
  heuristic ``β = 2·rho`` which produced ``corr ≈ rho/2`` for ``p = 0.1``.
  Saturates gracefully when the request exceeds what a logistic link can
  produce at the given baseline ``p``.

### Documentation

- README, `docs/index.md`, package docstring and ARCHITECTURE.md rewritten
  around concrete use-cases ("will CUPED be worth the effort?", "is my
  in-house t-test calibrated?", "delta-method or linearization for our
  CTR?", "is t-test safe on heavy-tailed revenue?") rather than abstract
  feature lists.

### Changed

- `_stats.welch_ttest` now returns the Welch–Satterthwaite degrees of freedom
  as a fifth tuple element so callers stop recomputing it.
- `Simulator.run()` now passes `batch_size="auto"` to `joblib.Parallel`,
  amortising per-task IPC overhead on fast criteria.

### Fixed

- `Bootstrap(method="bca")` previously reported a *percentile-style* p-value
  alongside a BCa CI; rejection now matches the BCa CI exactly via a
  BCa-adjusted ASL.
- `PostStratification` no longer drops small-`n` strata silently from the
  effect estimate; it falls back to the within-stratum pooled variance
  (Miratrix–Sekhon–Yu 2013, §3) for the variance contribution.
- The `__init__.py` doctest example now uses the real public API
  (`EffectSize`, `Simulator.seed`) instead of nonexistent constructor
  kwargs.
- `CUPAC._oof_predict_pooled` silences the actual `RuntimeWarning` numpy
  emits for degenerate LOO folds (the previous narrowing to
  `ConvergenceWarning` was incorrect — `RidgeCV` is closed-form and never
  emits `ConvergenceWarning`).
- Removed the unused `_stats.make_rng` helper.

## [0.1.0] — 2026-05-05

Initial release.

### Added

- `absim.Simulator` — Monte Carlo engine with `joblib`-based parallelism
  and reproducible per-iteration RNGs via `SeedSequence.spawn()`.
- Criteria:
  - `WelchTTest` — two-sample t-test with unequal variances.
  - `ZTestProportions` — z-test for difference of proportions.
  - `CUPED` — pre-experiment-covariate variance reduction.
  - `CUPAC` — out-of-fold pooled-model variant of CUPED.
  - `PostStratification` — closed-form stratified mean-difference test.
  - `PairedStratification` — paired t-test for genuinely paired samples
    (rank-matching fallback for unpaired data).
  - `Bootstrap` — non-parametric resampling with **percentile** and **BCa**
    intervals (jackknife acceleration).
  - `DeltaMethod` — Taylor-linearised variance for ratio metrics.
  - `Linearization` — Budylin per-unit linearisation for ratio metrics.
- Generators:
  - `ContinuousGenerator` — Gaussian / lognormal / mixture, optional pre-
    experiment covariate, optional paired sampling.
  - `BinaryGenerator` — Bernoulli outcomes with logistic-link covariate.
  - `RatioGenerator` — Poisson sessions × per-user rate with realistic
    numerator–denominator correlation.
- Reports — `SimulationReport` with Wilson 95% CI on the rejection rate;
  parquet + CSV IO; `plot_fpr_bar` and `plot_power_curve`.
- CLI — `absim run`, `absim list-criteria`, `absim list-experiments`.
- Hydra structured configs — `conf/{config,data,criterion,experiment}/*.yaml`.
- Pre-commit, ruff, mypy `--strict`, pytest with statistical tests.
- mkdocs + mkdocs-material + mkdocstrings documentation site.
- GitHub Actions CI matrix on Python 3.10 / 3.11 / 3.12.
