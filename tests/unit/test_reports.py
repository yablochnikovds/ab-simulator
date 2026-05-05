"""Unit tests for reports IO and plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pandas as pd
import pytest

from absim import EffectSize, Simulator
from absim.criteria import WelchTTest
from absim.generators import ContinuousGenerator
from absim.reports import (
    plot_fpr_bar,
    plot_power_curve,
    reports_to_dataframe,
    save_reports_parquet,
)


def _run(seed: int, effect: EffectSize) -> object:
    return Simulator(
        generator=ContinuousGenerator(n_per_group=20),
        criterion=WelchTTest(),
        n_sims=10,
        effect=effect,
        seed=seed,
    ).run(parallel=False)


def test_reports_to_dataframe_shape():
    reps = [_run(i, EffectSize("none", 0.0)) for i in range(3)]
    df = reports_to_dataframe(reps)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "rejection_rate" in df.columns


def test_save_reports_parquet(tmp_path: Path):
    reps = [_run(0, EffectSize("none", 0.0))]
    path = save_reports_parquet(reps, tmp_path / "out.parquet")
    assert path.exists()
    df = pd.read_parquet(path)
    assert df.iloc[0]["criterion"] == "welch_t"


def test_plot_fpr_bar_runs():
    reps = [_run(i, EffectSize("none", 0.0)) for i in range(2)]
    fig = plot_fpr_bar(reps)
    assert fig is not None


def test_plot_fpr_bar_rejects_h1_reports():
    reps = [_run(0, EffectSize("medium", 0.5))]
    with pytest.raises(ValueError, match="H₀"):
        plot_fpr_bar(reps)


def test_plot_power_curve_runs():
    reps = [
        _run(0, EffectSize("none", 0.0)),
        _run(1, EffectSize("small", 0.1)),
        _run(2, EffectSize("medium", 0.5)),
    ]
    fig = plot_power_curve(reps)
    assert fig is not None


def test_plot_fpr_bar_empty_raises():
    with pytest.raises(ValueError, match="no reports"):
        plot_fpr_bar([])


def test_plot_power_curve_empty_raises():
    with pytest.raises(ValueError, match="no reports"):
        plot_power_curve([])
