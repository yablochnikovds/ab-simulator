"""Hydra structured configs and config-store registration."""

from __future__ import annotations

from absim.config.schema import (
    BinaryDataConfig,
    BootstrapCriterionConfig,
    ContinuousDataConfig,
    CriterionConfig,
    CUPACCriterionConfig,
    CUPEDCriterionConfig,
    DataConfig,
    DeltaCriterionConfig,
    ExperimentConfig,
    LinearizationCriterionConfig,
    PairedCriterionConfig,
    PostStratCriterionConfig,
    RatioDataConfig,
    SimulatorConfig,
    WelchCriterionConfig,
    ZPropCriterionConfig,
    register_configs,
)

__all__ = [
    "BinaryDataConfig",
    "BootstrapCriterionConfig",
    "CUPACCriterionConfig",
    "CUPEDCriterionConfig",
    "ContinuousDataConfig",
    "CriterionConfig",
    "DataConfig",
    "DeltaCriterionConfig",
    "ExperimentConfig",
    "LinearizationCriterionConfig",
    "PairedCriterionConfig",
    "PostStratCriterionConfig",
    "RatioDataConfig",
    "SimulatorConfig",
    "WelchCriterionConfig",
    "ZPropCriterionConfig",
    "register_configs",
]
