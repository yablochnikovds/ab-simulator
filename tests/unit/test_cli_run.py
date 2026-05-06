"""End-to-end smoke test of the CLI's ``run`` subcommand.

Exercises ``_execute`` directly with a synthesized DictConfig, avoiding
Hydra's ``@hydra.main`` decorator (which we test indirectly by spawning
a subprocess in CI).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from omegaconf import OmegaConf

from absim.cli import _conf_dir, _execute


def _build_minimal_cfg(output_dir: Path):
    return OmegaConf.create(
        {
            "data": {
                "_target_": "absim.generators.ContinuousGenerator",
                "name": "continuous",
                "n_per_group": 30,
                "rho": 0.0,
            },
            "experiment": {
                "output_dir": str(output_dir),
                "effects": [
                    {"name": "none", "value": 0.0, "relative": False},
                    {"name": "small", "value": 0.1, "relative": False},
                ],
                "criteria": {
                    "welch": {
                        "_target_": "absim.criteria.WelchTTest",
                        "name": "welch_t",
                        "alpha": 0.05,
                    },
                },
            },
            "simulator": {
                "n_sims": 30,
                "alpha": 0.05,
                "seed": 0,
                "n_jobs": 1,
                "parallel": False,
            },
        }
    )


def test_execute_writes_artifacts(tmp_path: Path):
    cfg = _build_minimal_cfg(tmp_path)
    _execute(cfg)
    assert (tmp_path / "reports.parquet").exists()
    assert (tmp_path / "reports.csv").exists()
    assert (tmp_path / "fpr.png").exists()
    assert (tmp_path / "power.png").exists()


def test_conf_dir_resolves_to_existing_path():
    p = _conf_dir()
    # Must point at a directory, even when running from an installed wheel.
    assert isinstance(p, Path)
