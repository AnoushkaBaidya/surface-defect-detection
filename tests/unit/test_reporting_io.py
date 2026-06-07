from __future__ import annotations

from pathlib import Path

import pandas as pd

from defect_detection.reporting.io import write_dataframe_csv, write_markdown


def test_write_markdown_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "summary.md"
    write_markdown("# Test", output_path)
    assert output_path.read_text(encoding="utf-8") == "# Test"


def test_write_dataframe_csv_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "tables" / "metrics.csv"
    dataframe = pd.DataFrame([{"model_name": "patchcore", "recall": 0.98}])
    write_dataframe_csv(dataframe, output_path)
    loaded = pd.read_csv(output_path)
    assert loaded.iloc[0]["model_name"] == "patchcore"
