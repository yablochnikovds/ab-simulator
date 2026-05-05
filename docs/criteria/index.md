# Criterion reference

Each criterion is a small dataclass exposing a single
`test(treatment, control, **kwargs) -> TestResult` method.
Pages below give the formula, intuition, assumptions, and references.

| Page                                            | Family       | Notes                                       |
|-------------------------------------------------|--------------|---------------------------------------------|
| [Welch's t-test](welch.md)                      | continuous   | Unequal-variance two-sample t-test.         |
| [z-test for proportions](z_proportion.md)       | binary       | Pooled SE for the test, unpooled for CI.    |
| [CUPED](cuped.md)                               | continuous   | Variance reduction via pre-experiment cov.  |
| [CUPAC](cupac.md)                               | continuous   | CUPED with an out-of-fold ML predictor.     |
| [Bootstrap](bootstrap.md)                       | any          | Percentile + BCa with jackknife accel.      |
| [Delta-method](delta_method.md)                 | ratio        | Closed-form Taylor variance.                |
| [Linearization](linearization.md)               | ratio        | Budylin per-unit linearization.             |
| [Post-stratification](post_stratification.md)   | continuous   | Closed-form weighted mean-difference test.  |
| [Paired stratification](paired_stratification.md)| paired       | One-sample t-test on within-pair diffs.     |
