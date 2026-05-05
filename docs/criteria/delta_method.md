# Delta-method

For a ratio metric $\theta = \bar N / \bar D$ computed over per-unit
$(N_i, D_i)$ pairs, the Taylor expansion gives

$$
\widehat{\operatorname{Var}}(\hat\theta) \approx
    \frac{1}{\bar D^2} \widehat{\operatorname{Var}}(\bar N)
    - \frac{2 \bar N}{\bar D^3} \widehat{\operatorname{Cov}}(\bar N, \bar D)
    + \frac{\bar N^2}{\bar D^4} \widehat{\operatorname{Var}}(\bar D),
$$

and the test statistic for $\hat\theta_T - \hat\theta_C$ is the difference
divided by $\sqrt{\widehat{\operatorname{Var}}_T + \widehat{\operatorname{Var}}_C}$,
referred to a standard normal.

## When to prefer it

- Ratio metrics where the numerator and denominator are observed at
  different granularities (clicks per session, purchases per visit).
- When you want a closed-form CI/p-value without resampling.

## API

::: absim.criteria.DeltaMethod
    options:
      heading_level: 3

## Reference

Deng, A., Knoblich, U., & Lu, J. (2018). *Applying the Delta Method in
Metric Analytics: A Practical Guide with Novel Ideas.* KDD.
