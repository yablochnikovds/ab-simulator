# `absim.criteria`

The criterion package exposes a single `Criterion` Protocol plus a registry
helper API. Each shipped criterion has its own page under
[Criterion reference](../criteria/index.md), which doubles as the API
reference for that class.

## Protocol & registry

::: absim.criteria.base
    options:
      heading_level: 3
      show_root_heading: false
      show_submodules: false
      members:
        - Criterion
        - register
        - get
        - available

## Concrete criteria

| Class | Registry key | Page |
|---|---|---|
| `WelchTTest` | `welch_t` | [Welch's t-test](../criteria/welch.md) |
| `ZTestProportions` | `z_proportion` | [z-test for proportions](../criteria/z_proportion.md) |
| `CUPED` | `cuped` | [CUPED](../criteria/cuped.md) |
| `CUPAC` | `cupac` | [CUPAC](../criteria/cupac.md) |
| `Bootstrap` | `bootstrap` | [Bootstrap](../criteria/bootstrap.md) |
| `DeltaMethod` | `delta_method` | [Delta-method](../criteria/delta_method.md) |
| `Linearization` | `linearization` | [Linearization](../criteria/linearization.md) |
| `PostStratification` | `post_stratification` | [Post-stratification](../criteria/post_stratification.md) |
| `PairedStratification` | `paired_stratification` | [Paired stratification](../criteria/paired_stratification.md) |
