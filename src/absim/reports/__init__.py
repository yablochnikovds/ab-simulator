"""Aggregation, tabular IO, and plotting for simulator reports."""

from __future__ import annotations

from absim.reports.io import reports_to_dataframe, save_reports_parquet
from absim.reports.plot import plot_fpr_bar, plot_power_curve

__all__ = [
    "plot_fpr_bar",
    "plot_power_curve",
    "reports_to_dataframe",
    "save_reports_parquet",
]
