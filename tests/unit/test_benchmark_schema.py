from __future__ import annotations

from pathlib import Path

from defect_detection.evaluation.benchmark_schema import BenchmarkResult, BenchmarkRunConfig


def test_benchmark_result_to_dict_serializes_optional_fields() -> None:
    result = BenchmarkResult(
        run_id="run-1",
        model_family="cnn",
        model_name="resnet18",
        category="bottle",
        seed=42,
        training_mode="supervised",
        few_shot_k=None,
        classification_auroc=0.9,
    )

    payload = result.to_dict()
    assert payload["model_name"] == "resnet18"
    assert payload["classification_auroc"] == 0.9
    assert "segmentation_auroc" in payload


def test_benchmark_run_config_stores_paths() -> None:
    config = BenchmarkRunConfig(
        run_id="abc",
        model_family="anomalib",
        model_name="patchcore",
        category="screw",
        seed=42,
        training_mode="normal_only",
        supports_segmentation=True,
        few_shot_k=None,
        model_artifact_dir=Path("artifacts/models"),
        report_artifact_dir=Path("artifacts/reports"),
    )

    assert config.model_artifact_dir.name == "models"
