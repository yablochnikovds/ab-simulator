# `absim.generators`

The generator package exposes a `Generator` Protocol, a `Sample` dataclass,
and four concrete generators: one bootstrap-from-real-data
(`EmpiricalGenerator`) and three parametric stand-ins for when historical
data isn't available.

## Protocol & helpers

::: absim.generators.base
    options:
      heading_level: 3
      show_root_heading: false
      show_submodules: false
      members:
        - Generator
        - Sample
        - make_strata

## Bootstrap-from-real-data

::: absim.generators.empirical.EmpiricalGenerator
    options:
      heading_level: 3
      show_root_heading: true
      show_signature_annotations: true

## Parametric generators

::: absim.generators.continuous.ContinuousGenerator
    options:
      heading_level: 3
      show_root_heading: true
      show_signature_annotations: true

::: absim.generators.binary.BinaryGenerator
    options:
      heading_level: 3
      show_root_heading: true
      show_signature_annotations: true

::: absim.generators.ratio.RatioGenerator
    options:
      heading_level: 3
      show_root_heading: true
      show_signature_annotations: true
