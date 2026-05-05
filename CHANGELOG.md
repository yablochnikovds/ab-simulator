# Changelog

All notable changes to `absim` will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
