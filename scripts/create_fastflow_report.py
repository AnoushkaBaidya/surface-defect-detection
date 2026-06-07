"""
Create the FastFlow benchmark report.

Outputs:
- per_category_results.csv
- aggregate_results.csv
- fastflow_benchmark_report.md
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
    """Load FastFlow metric files into a dataframe."""
    rows = []

    for metrics_file in sorted(raw_metrics_dir.glob("*_fastflow_metrics.json")):
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
    """Create aggregate mean/std/min/max table."""
    rows = []

    for metric in METRIC_COLUMNS:
        if metric not in per_category_df.columns:
            continue

        values = per_category_df[metric].dropna().astype(float).tolist()

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


def create_markdown(per_category_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> str:
    """Create markdown report."""
    sorted_df = per_category_df.sort_values("classification_auroc", ascending=False)

    display_columns = [
        "category",
        "classification_auroc",
        "classification_f1",
        "recall",
        "false_negative_rate",
        "false_negative_count",
        "segmentation_auroc",
        "segmentation_f1",
        "training_time_seconds",
        "inference_latency_ms_mean",
        "throughput_images_per_second",
        "model_size_mb",
    ]

    available_display_columns = [
        column for column in display_columns if column in sorted_df.columns
    ]

    markdown = [
        "# FastFlow benchmark — FastFlow All-Category Benchmark Report",
        "",
        "## Purpose",
        "",
        "This report summarizes FastFlow normal-only anomaly detection performance across all 15 MVTec AD categories.",
        "",
        "FastFlow is evaluated as a trainable normalizing-flow anomaly detection model.",
        "",
        "## Training Protocol",
        "",
        "- Model: FastFlow",
        "- Model family: Anomalib",
        "- Training mode: normal-only trainable density model",
        "- Category-specific model: yes",
        "- Max epochs: 50",
        "- Early stopping: enabled when supported",
        "- Checkpointing: enabled when supported",
        "- Tuning focus: backbone, flow steps, hidden ratio, image size, localization quality, latency, and model size",
        "",
        "## Metric Notes",
        "",
        "- `classification_auroc` = image-level AUROC.",
        "- `segmentation_auroc` = pixel-level AUROC.",
        "- Recall/FNR are extracted from prediction labels when available.",
        "",
        "## Per-Category Results",
        "",
        sorted_df[available_display_columns].to_markdown(index=False),
        "",
        "## Aggregate Results",
        "",
        aggregate_df.to_markdown(index=False),
        "",
        "## Engineering Interpretation",
        "",
        "- FastFlow should be compared against PaDiM and PatchCore on accuracy, recall, localization, model size, and inference speed.",
        "- Because FastFlow is trainable, training time is a major operational factor.",
        "- If FastFlow approaches PatchCore performance with lower model size or latency, it may become a better production candidate.",
        "",
    ]

    return "\n".join(markdown)


def main() -> int:
    """Create FastFlow report."""
    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/fastflow_benchmark.yaml")

    raw_metrics_dir = PROJECT_ROOT / config["outputs"]["raw_metrics_dir"]
    per_category_csv = PROJECT_ROOT / config["outputs"]["per_category_results_csv"]
    aggregate_csv = PROJECT_ROOT / config["outputs"]["aggregate_results_csv"]
    markdown_report = PROJECT_ROOT / config["outputs"]["markdown_report"]

    per_category_df = load_metrics(raw_metrics_dir)

    if per_category_df.empty:
        raise FileNotFoundError(f"No FastFlow metrics found in {raw_metrics_dir}")

    aggregate_df = create_aggregate(per_category_df)

    per_category_csv.parent.mkdir(parents=True, exist_ok=True)

    per_category_df.to_csv(per_category_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)

    markdown_report.write_text(
        create_markdown(per_category_df, aggregate_df),
        encoding="utf-8",
    )

    logger.info("Saved per-category results: %s", per_category_csv)
    logger.info("Saved aggregate results: %s", aggregate_csv)
    logger.info("Saved markdown report: %s", markdown_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
