# Welch's t-test

Two-sample t-test for the difference of means under **unequal variances**.

## Statistic

$$
t = \frac{\bar Y_T - \bar Y_C}{\sqrt{s_T^2/n_T + s_C^2/n_C}},
\qquad
\nu = \frac{(s_T^2/n_T + s_C^2/n_C)^2}{(s_T^2/n_T)^2/(n_T-1) + (s_C^2/n_C)^2/(n_C-1)}.
$$

## Assumptions

- Independence within and between arms.
- Means are well-defined (i.e. finite-variance outcomes).
- Sample sizes are large enough for the t-distribution to be a good
  approximation (`n_per_group ≳ 30` is plenty for not-too-skewed data).

## When to prefer it

- Default for continuous metrics.
- Use as the comparison floor for variance-reduction methods (CUPED, post-strat).

## API

::: absim.criteria.WelchTTest
    options:
      heading_level: 3

## Reference

Welch, B. L. (1947). *The generalization of "Student's" problem when several
different population variances are involved.* Biometrika, 34(1/2), 28–35.
