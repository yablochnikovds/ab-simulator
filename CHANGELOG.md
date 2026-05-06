# Changelog

All notable changes to `absim` will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] — 2026-05-06

Polish release after the 0.2.0 real-data pivot. Closes a packaging bug that
broke the CLI for `pip`-installed users, restores `mkdocs --strict`
compatibility, and tightens documentation, tests, and the comparison table.

### Fixed

- **Packaging:** the Hydra `conf/` tree is now bundled into the wheel via
  `force-include` and located via `importlib.resources` at runtime. Without
  this, `absim run experiment=…` failed for users installed via PyPI / pip
  (the dev-checkout path was the only one that worked).
- **`mkdocs build --strict`** now passes — the bulk-render API pages
  (`docs/api/criteria.md`, `docs/api/generators.md`) no longer collide with
  per-class pages on `mkdocs_autorefs`. Without this, the gh-pages deploy
  workflow would have failed on the next push to `main`.
- **CONTRIBUTING.md** clone command typo (`yablochnikov` → `yablochnikovds`)
  fixed; new contributors no longer hit a 404 on the very first command.
- **`delta.py` references** corrected — the Hájek/Šidák/Sen *Theory of Rank
  Tests* citation was wrong for the delta method. Replaced with
  Casella & Berger §5.5.4 (textbook) and Deng et al. KDD 2018 (applied).
- **`Simulator`** is now `@dataclass(frozen=True, slots=True)` —
  consistent with criteria/generators and prevents accidental mid-flight
  mutation.
- Removed the unused `wilson_ci` import + dead `_binomial_ci_99` helper from
  `tests/statistical/test_fpr_power.py`.
- Deduplicated the `[0.2.0]` CHANGELOG section (previously had stray repeat
  blocks from a merge).

### Changed

- **British→American spelling** normalized across narrative text
  (`linearise → linearize`, `vectorised → vectorized`, `customise → customize`,
  `colour → color`, `synthesise → synthesize`, `visualisation → visualization`,
  `mis-calibrated → miscalibrated`, etc.). Class names, registry keys, and
  YAML keys were already American; the rot was confined to docstring prose.
- **Real-data tutorial Workflow 3** rewritten — previously titled "variance
  reduction on your CTR metric"; CUPED applies to *any* metric with a
  pre-experiment covariate (continuous, binary, ratio), not only to ratio
  metrics. The example now uses continuous revenue and notes how to extend
  to binary/ratio.
- **Comparison table in README** softened to honestly reflect
  `cluster_experiments` capabilities: it has CUPED but not CUPAC, and its
  bootstrap is for clustered designs, not real-data injection. CUPAC, BCa
  bootstrap, ratio-metric linearization, and the calibration-audit harness
  are the differentiators we keep highlighting.
- **`scripts/benchmark.py`** default `--n-sims=1000` matches the shipped
  `docs/benchmark.md` numbers; the help text spells out the trade-off
  ("raise to 5000 for tighter Wilson bands at 5x the runtime").
- **`pre-commit` hooks** bumped (`ruff` 0.7.4, `mypy` 1.13.0) to track CI.

### Added

- **Test coverage:** parametrized `BinaryGenerator` calibration across
  `p ∈ {0.1, 0.3, 0.5, 0.8}` plus a saturation test at
  unattainable `(p=0.05, rho=0.95)`. Direct unit tests for
  `_bca_two_sided_pvalue` (centred → p≈1; far-tail → p<1e-3; rejection
  matches `p_value < alpha`). A regression test that pooled-OOF CUPAC is
  unbiased for the ATE under H₁.

## [0.2.0] — 2026-05-06

Real-data simulation is the headline feature of this release: the simulator
now drives off your actual production distribution, not just three
parametric stand-ins.

### Added

- **`absim.generators.EmpiricalGenerator`** — bootstrap-resamples from real
  observed data (NumPy arrays / pandas columns from your warehouse) and
  injects calibrated effects into the treatment arm. Auto-detects
  continuous / binary / ratio metric types and supports paired covariates,
  strata, and numerator/denominator pairs. Closes the "but I can't run
  absim on my real data" objection: the simulator now drives off your
  actual distribution.
- New page **Power on real data** walking through three real-world
  workflows: power analysis on warehouse revenue, calibration audit on
  in-house statistical code, and CTR-with-CUPED comparison.
- `absim._stats.make_result(...)` and `absim._stats.degenerate_result(...)`
  helpers and a `absim._stats.two_sided_t_pvalue(...)` p-value helper that
  absorb the `TestResult` boilerplate from every criterion.
- `absim.generators.base.make_strata(...)` shared helper used by all
  parametric generators for equal-frequency stratum binning.

### Changed

- **`BinaryGenerator`** now numerically calibrates the logistic slope
  ``β`` (via Brent's method) so the realised ``corr(Y, X)`` actually
  matches the requested ``rho`` to ≈ 1e-3, instead of the previous loose
  heuristic ``β = 2·rho`` which produced ``corr ≈ rho/2`` for ``p = 0.1``.
  Saturates gracefully when the request exceeds what a logistic link can
  produce at the given baseline ``p``.
- `absim._stats.welch_ttest` now returns the Welch–Satterthwaite degrees of
  freedom as a fifth tuple element so callers stop recomputing it.
- `Simulator.run()` now passes `batch_size="auto"` to `joblib.Parallel`,
  amortizing per-task IPC overhead on fast criteria.

### Fixed

- `Bootstrap(method="bca")` previously reported a *percentile-style* p-value
  alongside a BCa CI; rejection now matches the BCa CI exactly via a
  BCa-adjusted achieved-significance level.
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
- Removed the unused `absim._stats.make_rng` helper.

### Documentation

- README, `docs/index.md`, package docstring and `ARCHITECTURE.md`
  rewritten around concrete real-data use-cases ("power-analyze a real
  metric from our warehouse", "is my in-house t-test calibrated?",
  "delta-method or linearization for our CTR?", "is t-test safe on
  heavy-tailed revenue?") rather than abstract feature lists.

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
  - `DeltaMethod` — Taylor-linearized variance for ratio metrics.
  - `Linearization` — Budylin per-unit linearization for ratio metrics.
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
