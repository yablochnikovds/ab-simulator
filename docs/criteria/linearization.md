# Linearization (Budylin)

Reduce the ratio $\bar N / \bar D$ to an *additive* per-unit metric

$$
L_i = \frac{N_i}{\bar D} - \frac{\bar N}{\bar D^2} D_i,
$$

then run any sample-mean test on $L$ (here Welch's t-test). To leading order
this is identical to the delta method, but lets you drop in **further
variance reduction** (CUPED, post-stratification) on top of $L$ — they
all just operate on a per-unit metric.

## Implementation note

`absim` linearizes against the **pooled** numerator and denominator means
(Budylin's recipe), so both arms share the same linear functional.

## API

::: absim.criteria.Linearization
    options:
      heading_level: 3

## Reference

Budylin, R. (2018). *Consistent transformation of ratio metrics for
efficient online controlled experiments.*
