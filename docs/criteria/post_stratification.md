# Post-stratification

Closed-form stratified estimator. Given strata indicators $s_i \in \{1, \ldots, K\}$,

$$
\hat\Delta = \sum_{k=1}^K w_k\, (\bar Y^T_k - \bar Y^C_k),
\qquad w_k = \frac{n_k}{n}
$$

with the **pooled-sample** weights $w_k$. Closed-form variance:

$$
\widehat{\operatorname{Var}}(\hat\Delta) = \sum_k w_k^2
    \left(\frac{s^2_{T,k}}{n_{T,k}} + \frac{s^2_{C,k}}{n_{C,k}}\right).
$$

The resulting test is referred to a Welch–Satterthwaite t-distribution.

## Why pooled weights?

If stratum sizes are estimated *per arm*, the difference contains a
random-weight term that increases variance. Using pooled weights — which
are independent of treatment assignment under randomisation — removes that
term. This is the *Miratrix–Sekhon–Yu* result; under stratified
randomisation it is *guaranteed* not to hurt vs. the unstratified Welch
test.

## API

::: absim.criteria.PostStratification
    options:
      heading_level: 3
