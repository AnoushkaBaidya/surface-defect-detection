"""
Benchmark schema definitions.

Purpose
-------
This module defines the standardized result format for all model families:
- CNN
- Anomalib
- VLM
- Hybrid

Why this matters
----------------
A senior ML benchmark must avoid ad-hoc result files. Every model/category run
should produce the same schema so results can be aggregated fairly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """
    Configuration for one benchmark run.

    Example:
    - model_family: anomalib
    - model_name: patchcore
    - category: bottle
    - seed: 42
    """

    run_id: str
    model_family: str
    model_name: str
    category: str
    seed: int
    training_mode: str
    supports_segmentation: bool
    few_shot_k: int | None
    model_artifact_dir: Path
    report_artifact_dir: Path


@dataclass
class BenchmarkResult:
    """
    Standardized benchmark result for one model/category/seed run.

    Metric naming convention
    ------------------------
    classification_* means image-level anomaly classification.
    segmentation_* means pixel-level localization/segmentation.

    Missing metrics should be stored as None, not omitted.
    """

    run_id: str
    model_family: str
    model_name: str
    category: str
    seed: int
    training_mode: str
    few_shot_k: int | None

    classification_auroc: float | None = None
    classification_f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    false_positive_rate: float | None = None
    false_negative_rate: float | None = None
    false_negative_count: int | None = None
    false_positive_count: int | None = None

    segmentation_auroc: float | None = None
    segmentation_f1: float | None = None
    segmentation_aupro: float | None = None

    training_time_seconds: float | None = None
    inference_latency_ms_mean: float | None = None
    inference_latency_ms_p95: float | None = None
    throughput_images_per_second: float | None = None
    model_size_mb: float | None = None

    normal_train_fraction: float | None = None
    defect_train_fraction: float | None = None

    model_artifact_path: str | None = None
    report_artifact_path: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON/CSV serialization."""
        return asdict(self)
