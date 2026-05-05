# Bootstrap

Non-parametric bootstrap for the difference of means, with **percentile**
and **BCa** confidence intervals.

## Algorithm

For $b = 1, \ldots, B$:

1. Resample treatment and control independently with replacement.
2. Compute $\hat\theta^*_b = \bar Y^*_T - \bar Y^*_C$.

The resulting bootstrap distribution gives the percentile interval directly.
The **BCa** interval applies bias correction $\hat z_0$ and skewness
correction $\hat a$ (estimated by jackknife) to the percentile endpoints:

$$
\hat a = \frac{\sum_i (\bar\theta_{(\cdot)} - \theta_{(i)})^3}
              {6 \left[\sum_i (\bar\theta_{(\cdot)} - \theta_{(i)})^2\right]^{3/2}},
$$

where $\theta_{(i)}$ are leave-one-out estimates.

## When to prefer it

- Heavy-tailed distributions where t-tests overstate confidence.
- Custom statistics where a closed-form variance is hard to derive
  (in this implementation we restrict to the difference of means).

## API

::: absim.criteria.Bootstrap
    options:
      heading_level: 3

## Reference

Efron, B., & Tibshirani, R. (1994). *An Introduction to the Bootstrap.*
Chapman & Hall.
