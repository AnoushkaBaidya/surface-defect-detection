"""
Create the ResNet18 benchmark report.

Purpose
-------
Aggregate category-specific ResNet18 JSON metrics into:
- per_category_results.csv
- aggregate_results.csv
- resnet18_benchmark_report.md
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
    "training_time_seconds",
    "inference_latency_ms_mean",
    "inference_latency_ms_p95",
    "throughput_images_per_second",
    "model_size_mb",
]


def load_result_files(raw_metrics_dir: Path) -> pd.DataFrame:
    """
    Load all category metrics JSON files.

    Parameters
    ----------
    raw_metrics_dir:
        Directory containing *_resnet18_metrics.json files.

    Returns
    -------
    pd.DataFrame
        Per-category results.
    """
    rows = []

    for metrics_file in sorted(raw_metrics_dir.glob("*_resnet18_metrics.json")):
        with metrics_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        row = {
            key: payload.get(key)
            for key in [
                "run_id",
                "model_family",
                "model_name",
                "category",
                "seed",
                "training_mode",
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

        rows.append(row)

    return pd.DataFrame(rows)


def create_aggregate_results(per_category_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate mean/std results for numeric metrics.

    Parameters
    ----------
    per_category_df:
        Per-category results.

    Returns
    -------
    pd.DataFrame
        Aggregate metric table.
    """
    rows = []

    for metric in METRIC_COLUMNS:
        values = [
            float(value) for value in per_category_df[metric].dropna().tolist() if value is not None
        ]

        if not values:
            continue

        rows.append(
            {
                "metric": metric,
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )

    return pd.DataFrame(rows)


def create_markdown_report(
    per_category_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
) -> str:
    """
    Create the markdown report.

    Parameters
    ----------
    per_category_df:
        Per-category result table.
    aggregate_df:
        Aggregate metric table.

    Returns
    -------
    str
        Markdown report.
    """
    sorted_df = per_category_df.sort_values("classification_auroc", ascending=False)

    worst_recall_df = per_category_df.sort_values("recall", ascending=True).head(5)
    worst_fnr_df = per_category_df.sort_values("false_negative_rate", ascending=False).head(5)

    markdown = [
        "# ResNet18 benchmark — ResNet18 All-Category Benchmark Report",
        "",
        "## Purpose",
        "",
        "This report summarizes the supervised ResNet18 CNN baseline across all 15 MVTec AD categories.",
        "",
        "The CNN baseline answers the question:",
        "",
        "> How well does a traditional supervised image classifier perform when only limited defect labels are available?",
        "",
        "## Training Protocol",
        "",
        "- Model: ResNet18",
        "- Pretraining: ImageNet pretrained",
        "- Training mode: category-specific supervised binary classification",
        "- Defect labels: limited fraction of test defects borrowed for controlled training",
        "- Early stopping: enabled",
        "- Class weighting: enabled",
        "- Scheduler: ReduceLROnPlateau",
        "- Primary industrial metric: recall / false-negative rate",
        "",
        "## Metric Applicability",
        "",
        "ResNet18 produces image-level predictions only. Therefore segmentation AUROC, segmentation F1, and segmentation AUPRO are not applicable.",
        "",
        "## Per-Category Results",
        "",
        sorted_df[
            [
                "category",
                "classification_auroc",
                "classification_f1",
                "precision",
                "recall",
                "false_negative_rate",
                "false_negative_count",
                "inference_latency_ms_mean",
                "throughput_images_per_second",
            ]
        ].to_markdown(index=False),
        "",
        "## Aggregate Results",
        "",
        aggregate_df.to_markdown(index=False),
        "",
        "## Lowest Recall Categories",
        "",
        worst_recall_df[
            [
                "category",
                "classification_auroc",
                "recall",
                "false_negative_rate",
                "false_negative_count",
            ]
        ].to_markdown(index=False),
        "",
        "## Highest False-Negative Rate Categories",
        "",
        worst_fnr_df[
            [
                "category",
                "classification_auroc",
                "recall",
                "false_negative_rate",
                "false_negative_count",
            ]
        ].to_markdown(index=False),
        "",
        "## Engineering Interpretation",
        "",
        "- High classification AUROC does not guarantee zero missed defects.",
        "- Recall and false-negative rate remain primary industrial safety/quality metrics.",
        "- This CNN baseline requires labeled defect examples, which are scarce in real manufacturing.",
        "- This baseline is intended for comparison against anomaly detection and VLM benchmarks.",
        "",
    ]

    return "\n".join(markdown)


def main() -> int:
    """Create ResNet18 report files."""
    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/resnet18_benchmark.yaml")

    raw_metrics_dir = PROJECT_ROOT / config["outputs"]["raw_metrics_dir"]
    per_category_csv = PROJECT_ROOT / config["outputs"]["per_category_results_csv"]
    aggregate_csv = PROJECT_ROOT / config["outputs"]["aggregate_results_csv"]
    markdown_report = PROJECT_ROOT / config["outputs"]["markdown_report"]

    per_category_csv.parent.mkdir(parents=True, exist_ok=True)

    per_category_df = load_result_files(raw_metrics_dir)

    if per_category_df.empty:
        raise FileNotFoundError(
            f"No ResNet18 metrics found in {raw_metrics_dir}. " "Run the ResNet18 benchmark first."
        )

    aggregate_df = create_aggregate_results(per_category_df)

    per_category_df.to_csv(per_category_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)

    report_text = create_markdown_report(per_category_df, aggregate_df)
    markdown_report.write_text(report_text, encoding="utf-8")

    logger.info("Saved per-category results: %s", per_category_csv)
    logger.info("Saved aggregate results: %s", aggregate_csv)
    logger.info("Saved markdown report: %s", markdown_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
