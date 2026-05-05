"""Hydra structured configs for the ``absim`` CLI.

These dataclasses mirror the runtime objects in :mod:`absim.criteria` and
:mod:`absim.generators`, plus a top-level :class:`ExperimentConfig` that
glues a generator, a list of criteria, and a list of effect sizes together.

The :func:`register_configs` function adds them to the Hydra
:class:`~hydra.core.config_store.ConfigStore`. Call it once at process start
(the CLI does this automatically).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

# --------------------------------------------------------------------------- #
# Criterion configs
# --------------------------------------------------------------------------- #


@dataclass
class CriterionConfig:
    """Base for every criterion config."""

    _target_: str = "???"
    name: str = "???"
    alpha: float = 0.05


@dataclass
class WelchCriterionConfig(CriterionConfig):
    _target_: str = "absim.criteria.WelchTTest"
    name: str = "welch_t"


@dataclass
class ZPropCriterionConfig(CriterionConfig):
    _target_: str = "absim.criteria.ZTestProportions"
    name: str = "z_proportion"


@dataclass
class CUPEDCriterionConfig(CriterionConfig):
    _target_: str = "absim.criteria.CUPED"
    name: str = "cuped"


@dataclass
class CUPACCriterionConfig(CriterionConfig):
    _target_: str = "absim.criteria.CUPAC"
    name: str = "cupac"
    n_splits: int = 5
    seed: int = 0


@dataclass
class BootstrapCriterionConfig(CriterionConfig):
    _target_: str = "absim.criteria.Bootstrap"
    name: str = "bootstrap"
    n_resamples: int = 2000
    method: str = "percentile"


@dataclass
class DeltaCriterionConfig(CriterionConfig):
    _target_: str = "absim.criteria.DeltaMethod"
    name: str = "delta_method"


@dataclass
class LinearizationCriterionConfig(CriterionConfig):
    _target_: str = "absim.criteria.Linearization"
    name: str = "linearization"


@dataclass
class PostStratCriterionConfig(CriterionConfig):
    _target_: str = "absim.criteria.PostStratification"
    name: str = "post_stratification"


@dataclass
class PairedCriterionConfig(CriterionConfig):
    _target_: str = "absim.criteria.PairedStratification"
    name: str = "paired_stratification"


# --------------------------------------------------------------------------- #
# Generator (data) configs
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Base config for synthetic data generators."""

    _target_: str = "???"
    name: str = "???"
    n_per_group: int = 1000


@dataclass
class ContinuousDataConfig(DataConfig):
    _target_: str = "absim.generators.ContinuousGenerator"
    name: str = "continuous"
    mean: float = 0.0
    sd: float = 1.0
    rho: float = 0.5
    distribution: str = "normal"
    n_strata: int = 4


@dataclass
class BinaryDataConfig(DataConfig):
    _target_: str = "absim.generators.BinaryGenerator"
    name: str = "binary"
    p: float = 0.1
    rho: float = 0.3
    n_strata: int = 4


@dataclass
class RatioDataConfig(DataConfig):
    _target_: str = "absim.generators.RatioGenerator"
    name: str = "ratio"
    base_rate: float = 0.2
    sessions_mean: float = 5.0
    noise: float = 0.1
    relative: bool = True
    n_strata: int = 4


# --------------------------------------------------------------------------- #
# Top-level configs
# --------------------------------------------------------------------------- #


@dataclass
class SimulatorConfig:
    """Top-level simulator runtime config."""

    n_sims: int = 10_000
    alpha: float = 0.05
    seed: int = 0
    n_jobs: int = -1
    parallel: bool = True


@dataclass
class EffectConfig:
    name: str = "none"
    value: float = 0.0
    relative: bool = False


@dataclass
class ExperimentConfig:
    """A full experiment specification: data × criteria × effects.

    Attributes
    ----------
    data
        Generator config.
    criteria
        Mapping ``criterion_name -> CriterionConfig`` to evaluate.
    effects
        List of effects to sweep (FPR + power curves).
    output_dir
        Where to save reports + plots, relative to Hydra's working dir.
    """

    data: DataConfig = field(default_factory=ContinuousDataConfig)
    criteria: dict[str, Any] = field(default_factory=dict)
    effects: list[EffectConfig] = field(
        default_factory=lambda: [
            EffectConfig(name="none", value=0.0),
            EffectConfig(name="small", value=0.05),
            EffectConfig(name="medium", value=0.1),
            EffectConfig(name="large", value=0.2),
        ]
    )
    output_dir: str = "outputs"


def register_configs() -> None:
    """Register all dataclass schemas with Hydra's :class:`ConfigStore`."""
    cs = ConfigStore.instance()
    cs.store(group="data", name="continuous", node=ContinuousDataConfig)
    cs.store(group="data", name="binary", node=BinaryDataConfig)
    cs.store(group="data", name="ratio", node=RatioDataConfig)
    cs.store(group="criterion", name="welch", node=WelchCriterionConfig)
    cs.store(group="criterion", name="z_proportion", node=ZPropCriterionConfig)
    cs.store(group="criterion", name="cuped", node=CUPEDCriterionConfig)
    cs.store(group="criterion", name="cupac", node=CUPACCriterionConfig)
    cs.store(group="criterion", name="bootstrap", node=BootstrapCriterionConfig)
    cs.store(group="criterion", name="delta", node=DeltaCriterionConfig)
    cs.store(group="criterion", name="linearization", node=LinearizationCriterionConfig)
    cs.store(group="criterion", name="post_stratification", node=PostStratCriterionConfig)
    cs.store(group="criterion", name="paired", node=PairedCriterionConfig)
    cs.store(name="simulator", node=SimulatorConfig)
    cs.store(name="experiment", node=ExperimentConfig)
