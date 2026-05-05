# CUPED

**Controlled-experiment Using Pre-Experiment Data.** Reduces variance by
removing the linear component of a pre-experiment covariate $X$ from the
outcome $Y$.

## Definition

$$
\theta = \frac{\operatorname{Cov}(Y, X)}{\operatorname{Var}(X)},
\qquad
Y_{\mathrm{adj}} = Y - \theta\, (X - \mathbb{E} X),
$$

then run Welch's t-test on $Y_{\mathrm{adj}}$. Asymptotic variance reduction
is

$$
\frac{\operatorname{Var}(Y_{\mathrm{adj}})}{\operatorname{Var}(Y)} = 1 - \rho^2,
$$

where $\rho = \operatorname{Corr}(Y, X)$.

## Assumptions

- $X$ is **independent of treatment assignment** (drawn before the experiment).
- The relationship $Y \mid X$ is well-approximated by a single linear slope
  on the pooled sample.

## API

::: absim.criteria.CUPED
    options:
      heading_level: 3

## Reference

Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). *Improving the sensitivity
of online controlled experiments by utilizing pre-experiment data.* WSDM '13.
