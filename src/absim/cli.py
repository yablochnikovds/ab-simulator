"""Console entry point for ``absim``.

Subcommands
-----------
- ``absim run experiment=<name> [overrides...]`` — Hydra-driven simulator run.
- ``absim list-criteria`` — print every registered criterion.
- ``absim list-experiments`` — print every shipped experiment YAML.

Implemented with :mod:`argparse` at the top level (so ``--help`` works without
Hydra's machinery) and Hydra inside the ``run`` subcommand for proper
config composition + override syntax.
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _list_criteria(_: argparse.Namespace) -> int:
    """List registered criteria."""
    from absim.criteria import available  # local import → faster CLI startup

    for name in available():
        print(name)
    return 0


def _conf_dir() -> Path:
    """Return the conf/ directory shipped with the source repo (best-effort)."""
    # Prefer in-repo conf/ when running from a checkout.
    here = Path(__file__).resolve()
    candidate = here.parent.parent.parent / "conf"
    if candidate.is_dir():
        return candidate
    # Fall back to package-included conf/ (we ship a copy via pyproject.toml).
    try:
        anchor = resources.files("absim").joinpath("../../conf")
        return Path(str(anchor)).resolve()
    except (FileNotFoundError, ModuleNotFoundError):
        return candidate


def _list_experiments(_: argparse.Namespace) -> int:
    """List experiment YAML names shipped in conf/experiment."""
    exp_dir = _conf_dir() / "experiment"
    if not exp_dir.is_dir():
        print(f"(no experiment dir found at {exp_dir})", file=sys.stderr)
        return 1
    for path in sorted(exp_dir.glob("*.yaml")):
        print(path.stem)
    return 0


def _run(ns: argparse.Namespace) -> int:
    """Hydra-compose & dispatch a simulation run."""
    # Build hydra arguments from the parsed argv tail.
    overrides: list[str] = list(ns.overrides)
    sys.argv = [sys.argv[0], *overrides]

    import hydra
    from omegaconf import DictConfig

    from absim.config import register_configs

    register_configs()

    config_path = str(_conf_dir().resolve())

    @hydra.main(version_base=None, config_path=config_path, config_name="config")
    def _entry(cfg: DictConfig) -> None:
        _execute(cfg)

    _entry()
    return 0


def _execute(cfg: Any) -> None:
    """Run the experiment described by ``cfg`` and persist outputs."""
    import time

    from hydra.utils import instantiate

    from absim import EffectSize, Simulator
    from absim.reports import (
        plot_fpr_bar,
        plot_power_curve,
        reports_to_dataframe,
        save_reports_parquet,
    )

    generator = instantiate(cfg.data)
    sim_cfg = cfg.simulator
    exp = cfg.experiment

    reports = []
    print(
        f"running experiment over {len(exp.criteria)} criteria × "
        f"{len(exp.effects)} effect(s) × {sim_cfg.n_sims} sims"
    )
    for label, crit_cfg in exp.criteria.items():
        criterion = instantiate(crit_cfg)
        # Tag each result with the user-chosen YAML key so duplicates of the
        # same underlying class (e.g. two Bootstrap configs) remain
        # distinguishable in reports.
        for effect_cfg in exp.effects:
            effect = EffectSize(
                name=effect_cfg.name,
                value=float(effect_cfg.value),
                relative=bool(effect_cfg.relative),
            )
            sim = Simulator(
                generator=generator,
                criterion=criterion,
                n_sims=int(sim_cfg.n_sims),
                alpha=float(sim_cfg.alpha),
                effect=effect,
                seed=int(sim_cfg.seed),
                n_jobs=int(sim_cfg.n_jobs),
            )
            t0 = time.perf_counter()
            report = sim.run(parallel=bool(sim_cfg.parallel))
            elapsed = time.perf_counter() - t0
            # Re-stamp criterion_name with the YAML key for unique reporting.
            from dataclasses import replace

            tagged = replace(report, criterion_name=label)
            reports.append(tagged)
            print(
                f"  {label:>14s} | {effect.name:>6s} | "
                f"rate={tagged.rejection_rate:.4f} "
                f"CI=[{tagged.binomial_ci_low:.3f},{tagged.binomial_ci_high:.3f}] "
                f"({elapsed:.2f}s)"
            )

    # Persist artifacts.
    out = Path(exp.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_path = save_reports_parquet(reports, out / "reports.parquet")
    df = reports_to_dataframe(reports)
    df.to_csv(out / "reports.csv", index=False)
    print(f"wrote {df_path}")

    # FPR plot (effect = 0 only) + power curve plot.
    h0 = [r for r in reports if r.effect.value == 0.0]
    if h0:
        fig = plot_fpr_bar(h0, alpha=float(sim_cfg.alpha))
        fpr_path = out / "fpr.png"
        fig.savefig(fpr_path, dpi=120)
        print(f"wrote {fpr_path}")
    if any(r.effect.value != 0.0 for r in reports):
        fig = plot_power_curve(reports)
        power_path = out / "power.png"
        fig.savefig(power_path, dpi=120)
        print(f"wrote {power_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="absim",
        description="A/B simulator: validate FPR ≈ α and quantify Power on synthetic data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an experiment (Hydra-driven)")
    p_run.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. experiment=continuous_welch_vs_cuped data.n_per_group=2000",
    )
    p_run.set_defaults(func=_run)

    p_lc = sub.add_parser("list-criteria", help="list registered criteria")
    p_lc.set_defaults(func=_list_criteria)

    p_le = sub.add_parser("list-experiments", help="list shipped experiments")
    p_le.set_defaults(func=_list_experiments)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
