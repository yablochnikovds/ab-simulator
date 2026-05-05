"""Coverage for the Hydra structured-config schema."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from absim.config import (
    BinaryDataConfig,
    BootstrapCriterionConfig,
    ContinuousDataConfig,
    CUPACCriterionConfig,
    CUPEDCriterionConfig,
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


def test_register_configs_populates_store():
    register_configs()
    cs = ConfigStore.instance()
    keys = cs.list("data")
    assert "continuous.yaml" in keys or any("continuous" in k for k in keys)
    crit_keys = cs.list("criterion")
    assert any("welch" in k for k in crit_keys)


def test_welch_config_instantiates_to_correct_class():
    cfg = OmegaConf.structured(WelchCriterionConfig())
    obj = instantiate(cfg)
    assert obj.name == "welch_t"
    assert obj.alpha == 0.05


def test_continuous_config_instantiates():
    cfg = OmegaConf.structured(ContinuousDataConfig(n_per_group=42, sd=2.0))
    obj = instantiate(cfg)
    assert obj.n_per_group == 42
    assert obj.sd == 2.0


def test_binary_config_instantiates():
    cfg = OmegaConf.structured(BinaryDataConfig(p=0.3))
    obj = instantiate(cfg)
    assert obj.p == 0.3


def test_ratio_config_instantiates():
    cfg = OmegaConf.structured(RatioDataConfig(base_rate=0.1))
    obj = instantiate(cfg)
    assert obj.base_rate == 0.1


def test_other_criterion_configs_instantiate():
    """Smoke test that every criterion config has a working _target_."""
    configs = [
        ZPropCriterionConfig(),
        CUPEDCriterionConfig(),
        CUPACCriterionConfig(),
        BootstrapCriterionConfig(),
        DeltaCriterionConfig(),
        LinearizationCriterionConfig(),
        PostStratCriterionConfig(),
        PairedCriterionConfig(),
    ]
    for cfg in configs:
        obj = instantiate(OmegaConf.structured(cfg))
        assert hasattr(obj, "name")


def test_experiment_config_has_defaults():
    cfg = ExperimentConfig()
    assert len(cfg.effects) == 4
    assert cfg.effects[0].name == "none"


def test_simulator_config_defaults():
    cfg = SimulatorConfig()
    assert cfg.alpha == 0.05
    assert cfg.parallel is True
