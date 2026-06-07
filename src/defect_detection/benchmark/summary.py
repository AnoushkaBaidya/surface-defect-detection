"""
Shared helpers for aggregate benchmark summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_model_metrics(model_name: str, metrics_dir: Path) -> pd.DataFrame:
    """Load metric JSON files for a single model family."""
    rows: list[dict[str, object]] = []

    if not metrics_dir.exists():
        return pd.DataFrame()

    for metrics_file in sorted(metrics_dir.glob("*.json")):
        with metrics_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        rows.append(
            {
                "model_name": model_name,
                "category": payload.get("category"),
                "classification_auroc": payload.get("classification_auroc"),
                "classification_f1": payload.get("classification_f1"),
                "precision": payload.get("precision"),
                "recall": payload.get("recall"),
                "false_negative_rate": payload.get("false_negative_rate"),
                "false_positive_rate": payload.get("false_positive_rate"),
                "false_negative_count": payload.get("false_negative_count"),
                "false_positive_count": payload.get("false_positive_count"),
                "segmentation_auroc": payload.get("segmentation_auroc"),
                "segmentation_f1": payload.get("segmentation_f1"),
                "segmentation_aupro": payload.get("segmentation_aupro"),
                "training_time_seconds": payload.get("training_time_seconds"),
                "inference_latency_ms_mean": payload.get("inference_latency_ms_mean"),
                "throughput_images_per_second": payload.get("throughput_images_per_second"),
                "model_size_mb": payload.get("model_size_mb"),
            }
        )

    return pd.DataFrame(rows)


def compute_aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean metrics grouped by model."""
    return (
        df.groupby("model_name")
        .agg(
            {
                "classification_auroc": "mean",
                "classification_f1": "mean",
                "precision": "mean",
                "recall": "mean",
                "false_negative_rate": "mean",
                "false_positive_rate": "mean",
                "segmentation_auroc": "mean",
                "segmentation_f1": "mean",
                "training_time_seconds": "mean",
                "inference_latency_ms_mean": "mean",
                "throughput_images_per_second": "mean",
                "model_size_mb": "mean",
            }
        )
        .reset_index()
    )


def compute_runtime_ranking(
    df: pd.DataFrame,
    recall_weight: float = 40.0,
    fnr_weight: float = 30.0,
    segmentation_weight: float = 15.0,
    latency_weight: float = 10.0,
    size_weight: float = 5.0,
) -> pd.DataFrame:
    """Compute a runtime-aware ranking score."""
    ranking_df = df.copy()

    ranking_df["recall_score"] = ranking_df["recall"] * recall_weight
    ranking_df["fnr_score"] = (1.0 - ranking_df["false_negative_rate"]) * fnr_weight
    ranking_df["segmentation_score"] = (
        ranking_df["segmentation_auroc"].fillna(0.0) * segmentation_weight
    )

    max_latency = ranking_df["inference_latency_ms_mean"].max()
    if max_latency and max_latency > 0:
        ranking_df["latency_score"] = (
            1.0 - (ranking_df["inference_latency_ms_mean"] / max_latency)
        ) * latency_weight
    else:
        ranking_df["latency_score"] = latency_weight

    max_model_size = ranking_df["model_size_mb"].max()
    if max_model_size and max_model_size > 0:
        ranking_df["model_size_score"] = (
            1.0 - (ranking_df["model_size_mb"] / max_model_size)
        ) * size_weight
    else:
        ranking_df["model_size_score"] = size_weight

    ranking_df["runtime_score"] = (
        ranking_df["recall_score"]
        + ranking_df["fnr_score"]
        + ranking_df["segmentation_score"]
        + ranking_df["latency_score"]
        + ranking_df["model_size_score"]
    )

    return ranking_df.sort_values("runtime_score", ascending=False)
