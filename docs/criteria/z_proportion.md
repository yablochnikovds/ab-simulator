# z-test for proportions

Two-sample z-test for the difference between Bernoulli probabilities.

## Statistic

$$
z = \frac{\hat p_T - \hat p_C}{\sqrt{\bar p (1 - \bar p)(1/n_T + 1/n_C)}},
$$

where $\bar p$ is the pooled proportion under H₀. The CI uses an *unpooled*
SE: $\sqrt{\hat p_T (1 - \hat p_T)/n_T + \hat p_C (1 - \hat p_C)/n_C}$.

## Assumptions

- Independent Bernoulli observations within each arm.
- $n_T \hat p_T (1 - \hat p_T)$ and the equivalent for control are large
  enough for the normal approximation (rule of thumb ≥ 5 successes and ≥ 5
  failures per arm).

## API

::: absim.criteria.ZTestProportions
    options:
      heading_level: 3
