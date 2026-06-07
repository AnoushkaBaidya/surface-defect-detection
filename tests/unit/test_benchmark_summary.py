from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from defect_detection.benchmark.summary import (
    compute_aggregate_metrics,
    compute_runtime_ranking,
    load_model_metrics,
)


def test_load_model_metrics_reads_json_payloads(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "bottle.json").write_text(
        json.dumps(
            {
                "category": "bottle",
                "classification_auroc": 0.9,
                "classification_f1": 0.8,
                "precision": 0.7,
                "recall": 0.95,
                "false_negative_rate": 0.05,
                "model_size_mb": 100.0,
            }
        ),
        encoding="utf-8",
    )

    df = load_model_metrics("patchcore", metrics_dir)
    assert list(df["model_name"]) == ["patchcore"]
    assert float(df.iloc[0]["recall"]) == 0.95


def test_compute_aggregate_metrics_groups_by_model() -> None:
    rows = [
        {
            "model_name": "a",
            "classification_auroc": 0.8,
            "classification_f1": 0.7,
            "precision": 0.6,
            "recall": 0.9,
            "false_negative_rate": 0.1,
            "false_positive_rate": 0.2,
            "segmentation_auroc": 0.0,
            "segmentation_f1": 0.0,
            "training_time_seconds": 1.0,
            "inference_latency_ms_mean": 2.0,
            "throughput_images_per_second": 3.0,
            "model_size_mb": 4.0,
        },
        {
            "model_name": "a",
            "classification_auroc": 1.0,
            "classification_f1": 0.9,
            "precision": 0.8,
            "recall": 1.0,
            "false_negative_rate": 0.0,
            "false_positive_rate": 0.1,
            "segmentation_auroc": 0.0,
            "segmentation_f1": 0.0,
            "training_time_seconds": 3.0,
            "inference_latency_ms_mean": 4.0,
            "throughput_images_per_second": 5.0,
            "model_size_mb": 6.0,
        },
    ]
    df = pd.DataFrame(rows)

    aggregate = compute_aggregate_metrics(df)
    assert aggregate.shape[0] == 1
    assert float(aggregate.iloc[0]["classification_auroc"]) == 0.9


def test_compute_runtime_ranking_prefers_higher_recall_and_lower_latency() -> None:
    df = pd.DataFrame(
        [
            {
                "model_name": "fast",
                "recall": 0.95,
                "false_negative_rate": 0.05,
                "segmentation_auroc": 0.9,
                "inference_latency_ms_mean": 100.0,
                "model_size_mb": 100.0,
            },
            {
                "model_name": "slow",
                "recall": 0.90,
                "false_negative_rate": 0.10,
                "segmentation_auroc": 0.9,
                "inference_latency_ms_mean": 1000.0,
                "model_size_mb": 500.0,
            },
        ]
    )

    ranking = compute_runtime_ranking(df)
    assert ranking.iloc[0]["model_name"] == "fast"
