# Paired stratification

Paired t-test on within-pair differences $d_i = T_i - C_i$.

## When to use it

`PairedStratification(paired=True)` (default) requires **genuinely paired
data** — for example before/after measurements, twin studies, or the
output of a paired generator. In that case the differences are i.i.d. and
the classical paired-t standard error is correct.

For independent A/B samples, paired matching by covariate rank
(`paired=False`) is **approximate** — see the architecture notes for the
small FPR inflation it introduces. We recommend using CUPED or
`PostStratification` for unpaired data, and reserving the paired test for
the cases where pairing is real.

## API

::: absim.criteria.PairedStratification
    options:
      heading_level: 3
