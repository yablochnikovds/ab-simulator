"""Unit tests for the simulator engine."""

from __future__ import annotations

import numpy as np
import pytest

from absim import EffectSize, Simulator
from absim.criteria import WelchTTest
from absim.generators import ContinuousGenerator


def test_simulator_runs_serially_and_returns_report():
    sim = Simulator(
        generator=ContinuousGenerator(n_per_group=50),
        criterion=WelchTTest(),
        n_sims=20,
        alpha=0.05,
        seed=0,
    )
    rep = sim.run(parallel=False)
    assert rep.n_sims == 20
    assert rep.criterion_name == "welch_t"
    assert 0.0 <= rep.rejection_rate <= 1.0
    assert rep.binomial_ci_low <= rep.rejection_rate <= rep.binomial_ci_high


def test_simulator_parallel_matches_serial():
    """Parallel and serial runs must produce identical reports — RNGs are spawned per iteration."""
    common = dict(
        generator=ContinuousGenerator(n_per_group=50),
        criterion=WelchTTest(),
        n_sims=30,
        alpha=0.05,
        seed=42,
    )
    rep_serial = Simulator(**common).run(parallel=False)
    rep_parallel = Simulator(**common, n_jobs=2).run(parallel=True)
    assert rep_serial.rejection_rate == rep_parallel.rejection_rate
    assert rep_serial.mean_pvalue == pytest.approx(rep_parallel.mean_pvalue)
    assert rep_serial.mean_effect == pytest.approx(rep_parallel.mean_effect)


def test_simulator_under_h1_higher_rejection():
    h0 = Simulator(
        generator=ContinuousGenerator(n_per_group=200),
        criterion=WelchTTest(),
        n_sims=100,
        seed=0,
    ).run(parallel=False)
    h1 = Simulator(
        generator=ContinuousGenerator(n_per_group=200),
        criterion=WelchTTest(),
        n_sims=100,
        effect=EffectSize(name="big", value=0.5),
        seed=0,
    ).run(parallel=False)
    assert h1.rejection_rate > h0.rejection_rate


def test_simulator_rejects_invalid_n_sims():
    sim = Simulator(
        generator=ContinuousGenerator(n_per_group=10),
        criterion=WelchTTest(),
        n_sims=0,
    )
    with pytest.raises(ValueError, match="positive"):
        sim.run()


def test_simulator_report_to_dict_round_trip():
    sim = Simulator(
        generator=ContinuousGenerator(n_per_group=20),
        criterion=WelchTTest(),
        n_sims=5,
        seed=0,
    )
    rep = sim.run(parallel=False)
    d = rep.to_dict()
    assert d["criterion"] == "welch_t"
    assert d["n_sims"] == 5
    assert "ci_low" in d
    assert "rejection_rate" in d


def test_simulation_report_aliases():
    sim = Simulator(
        generator=ContinuousGenerator(n_per_group=20),
        criterion=WelchTTest(),
        n_sims=5,
        seed=0,
    )
    rep = sim.run(parallel=False)
    assert rep.fpr == rep.rejection_rate
    assert rep.power == rep.rejection_rate


def test_simulator_reproducible_across_runs():
    sim_a = Simulator(
        generator=ContinuousGenerator(n_per_group=50),
        criterion=WelchTTest(),
        n_sims=20,
        seed=7,
    )
    sim_b = Simulator(
        generator=ContinuousGenerator(n_per_group=50),
        criterion=WelchTTest(),
        n_sims=20,
        seed=7,
    )
    a = sim_a.run(parallel=False)
    b = sim_b.run(parallel=False)
    assert a.rejection_rate == b.rejection_rate
    assert a.mean_pvalue == b.mean_pvalue


def test_seed_split_independent_iterations():
    """Two iterations must produce different sample-level outputs."""
    from absim.simulator import _run_one

    gen = ContinuousGenerator(n_per_group=30)
    crit = WelchTTest()
    seeds = np.random.SeedSequence(0).spawn(2)
    r1 = _run_one(gen, crit, seeds[0], 0.0)
    r2 = _run_one(gen, crit, seeds[1], 0.0)
    assert r1.effect != r2.effect
