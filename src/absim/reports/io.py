"""Tabular IO for simulation reports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from absim.types import SimulationReport


def reports_to_dataframe(reports: Iterable[SimulationReport]) -> pd.DataFrame:
    """Flatten a sequence of :class:`SimulationReport` into a tidy DataFrame."""
    return pd.DataFrame([r.to_dict() for r in reports])


def save_reports_parquet(reports: Iterable[SimulationReport], path: str | Path) -> Path:
    """Persist reports as a parquet file. Returns the resolved output path."""
    df = reports_to_dataframe(reports)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
