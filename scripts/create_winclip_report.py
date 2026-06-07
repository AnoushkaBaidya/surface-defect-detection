"""
Create the WinCLIP benchmark report.

Outputs:
- per_category_results.csv
- aggregate_results.csv
- winclip_benchmark_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.utils.config import load_yaml_config  # noqa: E402
from defect_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


METRIC_COLUMNS = [
    "classification_auroc",
    "classification_f1",
    "precision",
    "recall",
    "false_positive_rate",
    "false_negative_rate",
    "false_negative_count",
    "false_positive_count",
    "segmentation_auroc",
    "segmentation_f1",
    "segmentation_aupro",
    "training_time_seconds",
    "inference_latency_ms_mean",
    "inference_latency_ms_p95",
    "throughput_images_per_second",
    "model_size_mb",
]


def load_metrics(raw_metrics_dir: Path) -> pd.DataFrame:
    """Load all WinCLIP metric JSON files."""
    rows = []

    for metrics_file in sorted(raw_metrics_dir.rglob("*_winclip_k*_metrics.json")):
        with metrics_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        rows.append(
            {
                key: payload.get(key)
                for key in [
                    "run_id",
                    "model_family",
                    "model_name",
                    "category",
                    "seed",
                    "training_mode",
                    "few_shot_k",
                    "classification_auroc",
                    "classification_f1",
                    "precision",
                    "recall",
                    "false_positive_rate",
                    "false_negative_rate",
                    "false_negative_count",
                    "false_positive_count",
                    "segmentation_auroc",
                    "segmentation_f1",
                    "segmentation_aupro",
                    "training_time_seconds",
                    "inference_latency_ms_mean",
                    "inference_latency_ms_p95",
                    "throughput_images_per_second",
                    "model_size_mb",
                    "normal_train_fraction",
                    "defect_train_fraction",
                    "model_artifact_path",
                    "notes",
                ]
            }
        )

    return pd.DataFrame(rows)


def create_aggregate(per_category_df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate table grouped by k-shot."""
    rows = []

    for k_shot, group_df in per_category_df.groupby("few_shot_k"):
        for metric in METRIC_COLUMNS:
            if metric not in group_df.columns:
                continue

            values = group_df[metric].dropna().astype(float).tolist()
            if not values:
                continue

            rows.append(
                {
                    "model_name": "winclip",
                    "few_shot_k": int(k_shot),
                    "metric": metric,
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )

    return pd.DataFrame(rows)


def create_wide_aggregate(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    """Create wide aggregate table for easier comparison."""
    if aggregate_df.empty:
        return aggregate_df

    return aggregate_df.pivot_table(
        index=["model_name", "few_shot_k"],
        columns="metric",
        values="mean",
    ).reset_index()


def create_markdown(per_category_df: pd.DataFrame, aggregate_wide_df: pd.DataFrame) -> str:
    """Create markdown report."""
    display_columns = [
        "category",
        "few_shot_k",
        "classification_auroc",
        "classification_f1",
        "recall",
        "false_negative_rate",
        "false_negative_count",
        "segmentation_auroc",
        "segmentation_f1",
        "inference_latency_ms_mean",
        "throughput_images_per_second",
    ]

    available_display_columns = [
        column for column in display_columns if column in per_category_df.columns
    ]

    sorted_df = per_category_df.sort_values(
        ["few_shot_k", "classification_auroc"],
        ascending=[True, False],
    )

    markdown = [
        "# WinCLIP benchmark — WinCLIP Zero/Few-Shot Benchmark Report",
        "",
        "## Purpose",
        "",
        "This report summarizes WinCLIP zero-shot and few-shot anomaly detection across all MVTec AD categories.",
        "",
        "WinCLIP evaluates whether CLIP-based language/image representations can detect industrial anomalies without category-specific training.",
        "",
        "## Protocol",
        "",
        "- Model: WinCLIP",
        "- Model family: VLM",
        "- Training mode: zero-shot or few-shot",
        "- k-shot settings: 0, 1, 4",
        "- Category-specific learned weights: no",
        "- Category-specific prompts/reference images: yes",
        "",
        "## Metric Notes",
        "",
        "- `classification_auroc` = image-level AUROC.",
        "- `segmentation_auroc` = pixel-level AUROC if available.",
        "- k=0 means zero-shot; k>0 means few-shot normal references.",
        "",
        "## Aggregate Results by k-shot",
        "",
        aggregate_wide_df.to_markdown(index=False),
        "",
        "## Per-Category Results",
        "",
        sorted_df[available_display_columns].to_markdown(index=False),
        "",
        "## Engineering Interpretation",
        "",
        "- WinCLIP should not be expected to always beat PatchCore/FastFlow.",
        "- Its value is zero/few-shot adaptability and reduced category-specific training.",
        "- Strong performance here supports deployment scenarios with new products and limited training data.",
        "- Weak performance still provides useful evidence that domain-specific anomaly models are required.",
        "",
    ]

    return "\n".join(markdown)


def main() -> int:
    """Create the WinCLIP benchmark report."""
    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/winclip_benchmark.yaml")

    raw_metrics_dir = PROJECT_ROOT / config["outputs"]["raw_metrics_dir"]
    per_category_csv = PROJECT_ROOT / config["outputs"]["per_category_results_csv"]
    aggregate_csv = PROJECT_ROOT / config["outputs"]["aggregate_results_csv"]
    markdown_report = PROJECT_ROOT / config["outputs"]["markdown_report"]

    per_category_df = load_metrics(raw_metrics_dir)

    if per_category_df.empty:
        raise FileNotFoundError(f"No WinCLIP metrics found in {raw_metrics_dir}")

    aggregate_long_df = create_aggregate(per_category_df)
    aggregate_wide_df = create_wide_aggregate(aggregate_long_df)

    per_category_csv.parent.mkdir(parents=True, exist_ok=True)

    per_category_df.to_csv(per_category_csv, index=False)
    aggregate_wide_df.to_csv(aggregate_csv, index=False)

    markdown_report.write_text(
        create_markdown(per_category_df, aggregate_wide_df),
        encoding="utf-8",
    )

    logger.info("Saved per-category results: %s", per_category_csv)
    logger.info("Saved aggregate results: %s", aggregate_csv)
    logger.info("Saved markdown report: %s", markdown_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
