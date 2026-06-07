"""Helpers for persisting benchmark reports and tabular outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_dataframe_csv(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    """Write a dataframe to CSV, creating parent directories when needed."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)


def write_markdown(content: str, output_path: str | Path) -> None:
    """Write markdown content to disk, creating parent directories when needed."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
