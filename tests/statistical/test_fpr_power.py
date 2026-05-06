"""End-to-end statistical validation: FPR ≈ alpha and Power monotone in effect.

These tests run real Monte Carlo simulations. They are somewhat slow (a few
seconds each) and are marked ``statistical`` so the full suite can opt in /
out via ``pytest -m statistical`` or ``pytest -m 'not statistical'``.

We validate FPR with a 99% binomial CI: at α = 0.05 and N = 3000 sims, the
CI is approximately [0.040, 0.060] under H₀. A failure here means the
criterion is genuinely miscalibrated, not flaky.
"""

from __future__ import annotations

import pytest
from scipy.stats import binom

from absim import EffectSize, Simulator
from absim.criteria import (
    CUPED,
    Bootstrap,
    DeltaMethod,
    Linearization,
    PairedStratification,
    PostStratification,
    WelchTTest,
    ZTestProportions,
)
from absim.generators import BinaryGenerator, ContinuousGenerator, RatioGenerator

pytestmark = pytest.mark.statistical


N_SIMS = 3000
ALPHA = 0.05
SEED = 13


def _assert_fpr_in_ci(rate: float, n_sims: int, alpha: float = ALPHA) -> None:
    # Compare the *expected* alpha against the 99% CI of the *observed* rate.
    # We reject the criterion if `alpha` is outside that CI.
    rejections = int(round(rate * n_sims))
    p_low = binom.ppf(0.005, n_sims, alpha) / n_sims
    p_high = binom.ppf(0.995, n_sims, alpha) / n_sims
    assert p_low <= rate <= p_high, (
        f"observed FPR {rate:.4f} (rejections={rejections}/{n_sims}) "
        f"outside 99% binomial CI [{p_low:.4f}, {p_high:.4f}] of α={alpha}"
    )


# --------------------------------------------------------------------------- #
# FPR (H0)
# --------------------------------------------------------------------------- #


def test_welch_fpr_calibrated_continuous():
    sim = Simulator(
        generator=ContinuousGenerator(n_per_group=200, rho=0.0),
        criterion=WelchTTest(),
        n_sims=N_SIMS,
        alpha=ALPHA,
        seed=SEED,
    )
    rep = sim.run()
    _assert_fpr_in_ci(rep.rejection_rate, N_SIMS)


def test_cuped_fpr_calibrated_continuous():
    sim = Simulator(
        generator=ContinuousGenerator(n_per_group=200, rho=0.6),
        criterion=CUPED(),
        n_sims=N_SIMS,
        alpha=ALPHA,
        seed=SEED,
    )
    rep = sim.run()
    _assert_fpr_in_ci(rep.rejection_rate, N_SIMS)


def test_post_strat_fpr_calibrated_continuous():
    sim = Simulator(
        generator=ContinuousGenerator(n_per_group=300, rho=0.5, n_strata=4),
        criterion=PostStratification(),
        n_sims=N_SIMS,
        alpha=ALPHA,
        seed=SEED,
    )
    rep = sim.run()
    _assert_fpr_in_ci(rep.rejection_rate, N_SIMS)


def test_paired_fpr_calibrated_paired_data():
    sim = Simulator(
        generator=ContinuousGenerator(n_per_group=300, rho=0.6, paired=True),
        criterion=PairedStratification(paired=True),
        n_sims=N_SIMS,
        alpha=ALPHA,
        seed=SEED,
    )
    rep = sim.run()
    _assert_fpr_in_ci(rep.rejection_rate, N_SIMS)


def test_z_proportion_fpr_calibrated():
    sim = Simulator(
        generator=BinaryGenerator(n_per_group=2000, p=0.1, rho=0.0),
        criterion=ZTestProportions(),
        n_sims=N_SIMS,
        alpha=ALPHA,
        seed=SEED,
    )
    rep = sim.run()
    _assert_fpr_in_ci(rep.rejection_rate, N_SIMS)


def test_delta_method_fpr_calibrated_ratio():
    sim = Simulator(
        generator=RatioGenerator(n_per_group=500),
        criterion=DeltaMethod(),
        n_sims=N_SIMS,
        alpha=ALPHA,
        seed=SEED,
    )
    rep = sim.run()
    _assert_fpr_in_ci(rep.rejection_rate, N_SIMS)


def test_linearization_fpr_calibrated_ratio():
    sim = Simulator(
        generator=RatioGenerator(n_per_group=500),
        criterion=Linearization(),
        n_sims=N_SIMS,
        alpha=ALPHA,
        seed=SEED,
    )
    rep = sim.run()
    _assert_fpr_in_ci(rep.rejection_rate, N_SIMS)


def test_bootstrap_percentile_fpr_calibrated():
    sim = Simulator(
        generator=ContinuousGenerator(n_per_group=200, rho=0.0),
        criterion=Bootstrap(n_resamples=500, method="percentile"),
        n_sims=1500,
        alpha=ALPHA,
        seed=SEED,
    )
    rep = sim.run()
    # Bootstrap is approximate at small N; use a slightly looser gate.
    p_low = binom.ppf(0.005, 1500, ALPHA) / 1500
    p_high = binom.ppf(0.995, 1500, ALPHA) / 1500
    assert p_low - 0.01 <= rep.rejection_rate <= p_high + 0.01


# --------------------------------------------------------------------------- #
# Power monotonicity (H1)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "criterion",
    [
        WelchTTest(),
        CUPED(),
    ],
    ids=["welch", "cuped"],
)
def test_power_monotone_in_effect(criterion):
    """Larger effect ⇒ at-least-as-large power."""
    powers = []
    for eff in (0.0, 0.05, 0.1, 0.2):
        sim = Simulator(
            generator=ContinuousGenerator(n_per_group=200, rho=0.5),
            criterion=criterion,
            n_sims=1500,
            alpha=ALPHA,
            effect=EffectSize(name=str(eff), value=eff),
            seed=SEED,
        )
        powers.append(sim.run().rejection_rate)
    # Strictly non-decreasing within sampling noise (allow up to 0.02 wiggle).
    for prev, cur in zip(powers[:-1], powers[1:], strict=True):
        assert cur >= prev - 0.02


def test_cuped_power_at_least_welch_under_informative_covariate():
    common = dict(
        generator=ContinuousGenerator(n_per_group=300, rho=0.7),
        n_sims=2000,
        alpha=ALPHA,
        effect=EffectSize(name="medium", value=0.1),
        seed=SEED,
    )
    welch_pow = Simulator(criterion=WelchTTest(), **common).run().rejection_rate
    cuped_pow = Simulator(criterion=CUPED(), **common).run().rejection_rate
    # Allow tiny noise; expect a clear improvement.
    assert cuped_pow >= welch_pow + 0.05


def test_post_strat_power_at_least_welch_with_strata():
    common = dict(
        generator=ContinuousGenerator(n_per_group=400, rho=0.7, n_strata=4),
        n_sims=2000,
        alpha=ALPHA,
        effect=EffectSize(name="medium", value=0.1),
        seed=SEED,
    )
    welch_pow = Simulator(criterion=WelchTTest(), **common).run().rejection_rate
    ps_pow = Simulator(criterion=PostStratification(), **common).run().rejection_rate
    assert ps_pow >= welch_pow - 0.02
