"""Static plotting for simulation reports.

Both functions return the underlying :class:`~matplotlib.figure.Figure` so
callers can save / further customise. Seaborn supplies the colour palette but
all annotations are done with plain matplotlib for portability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from absim.reports.io import reports_to_dataframe

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.figure import Figure

    from absim.types import SimulationReport


def plot_fpr_bar(
    reports: Iterable[SimulationReport],
    *,
    alpha: float = 0.05,
    title: str = "False positive rate by criterion",
) -> Figure:
    """Bar chart of FPR with binomial CI error bars and a horizontal line at α.

    Parameters
    ----------
    reports
        Reports run under H₀ (effect = 0).
    alpha
        The nominal significance level — drawn as a reference line.
    title
        Plot title.
    """
    df = reports_to_dataframe(reports)
    if df.empty:
        raise ValueError("plot_fpr_bar received no reports")
    if (df["effect_value"] != 0).any():
        raise ValueError("plot_fpr_bar expects H₀ reports (effect_value == 0)")
    df = df.sort_values("rejection_rate")
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.4 * len(df)))
    palette = sns.color_palette("crest", n_colors=len(df))
    err_low = df["rejection_rate"] - df["ci_low"]
    err_high = df["ci_high"] - df["rejection_rate"]
    ax.barh(
        df["criterion"],
        df["rejection_rate"],
        xerr=[err_low, err_high],
        color=palette,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(alpha, ls="--", color="crimson", label=f"α = {alpha}")
    ax.set_xlabel("Rejection rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_power_curve(
    reports: Iterable[SimulationReport],
    *,
    title: str = "Power vs effect size",
) -> Figure:
    """Line plot of power vs effect size, one line per criterion."""
    df = reports_to_dataframe(reports)
    if df.empty:
        raise ValueError("plot_power_curve received no reports")
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("flare", n_colors=df["criterion"].nunique())
    for color, (criterion, sub) in zip(palette, df.groupby("criterion"), strict=False):
        sub = sub.sort_values("effect_value")
        ax.plot(
            sub["effect_value"],
            sub["rejection_rate"],
            marker="o",
            label=criterion,
            color=color,
        )
        ax.fill_between(
            sub["effect_value"],
            sub["ci_low"],
            sub["ci_high"],
            color=color,
            alpha=0.15,
        )
    ax.set_xlabel("Effect size")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.05, color="grey", lw=0.8, ls=":")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    # Encourage downstream callers to manage figure life-cycle.
    _ = np.array(0)  # keep numpy import non-noop for future extensions
    return fig
