# CUPAC

CUPED with a model-based covariate. Replaces the raw covariate $X$ with the
out-of-fold prediction $\hat f(X)$ of a regression of $Y$ on $X$, fit on
the **pooled** treatment + control sample.

## Why pooled, not per-arm?

Fitting a separate model per arm leaks the treatment effect into the
predictor — under H₁ the predictor for the treatment arm absorbs the very
shift you're trying to measure, and CUPED's adjustment kills the power.
Pooling fits one symmetric regression of $Y$ on $X$, preserves the average
treatment effect, and reduces only the within-arm variance.

## Inputs

The criterion accepts either:

- **`prediction_treatment` + `prediction_control`** — pre-computed
  predictions, the typical production deployment (model trained offline on
  pre-experiment data).
- **`features_treatment` + `features_control`** — feature matrices, in
  which case `absim` fits a `RidgeCV` with K-fold OOF predictions.

## API

::: absim.criteria.CUPAC
    options:
      heading_level: 3
