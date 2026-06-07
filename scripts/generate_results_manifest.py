from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import REPO_ROOT, file_sha256, load_yaml, write_json


def describe_file(path: Path) -> dict:
    return {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "exists": path.exists(),
        "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def run(config_path: str = "configs/pipeline/evaluate_resnet18_retained_sample.yaml") -> Path:
    config = load_yaml(REPO_ROOT / config_path)
    manifest = {
        "retained_sample_pipeline": {
            "dataset_contract": describe_file(REPO_ROOT / "docs/DATASET_CONTRACT.md"),
            "dataset_manifest_schema": describe_file(
                REPO_ROOT / "configs/pipeline/retained_sample_manifest.schema.json"
            ),
            "dataset_manifest_summary": describe_file(
                REPO_ROOT / "reports/datasets/retained_sample_manifest_summary.json"
            ),
            "split_summary": describe_file(
                REPO_ROOT
                / "reports/datasets/splits/retained_sample_oct14_to_nov8/split_summary.csv"
            ),
            "prediction_csv": describe_file(REPO_ROOT / config["outputs"]["prediction_csv"]),
            "threshold_summary_json": describe_file(
                REPO_ROOT / config["outputs"]["threshold_summary_json"]
            ),
            "metrics_json": describe_file(REPO_ROOT / config["outputs"]["metrics_json"]),
            "score_monitoring_json": describe_file(
                REPO_ROOT / config["outputs"]["score_monitoring_json"]
            ),
            "markdown_summary": describe_file(REPO_ROOT / config["outputs"]["markdown_summary"]),
            "model_artifact": describe_file(REPO_ROOT / config["model"]["artifact_path"]),
        },
        "selected_policy": {
            "service_config": describe_file(REPO_ROOT / "configs/final_model_policy.yaml"),
            "resnet18_onnx": describe_file(REPO_ROOT / "models/resnet18_internal.onnx"),
            "efficientad_onnx": describe_file(REPO_ROOT / "models/efficientad_score.onnx"),
            "efficientad_calibration": describe_file(
                REPO_ROOT / "configs/efficientad_calibration.json"
            ),
        },
    }

    output_path = REPO_ROOT / "reports/results_manifest.json"
    write_json(output_path, manifest)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pipeline/evaluate_resnet18_retained_sample.yaml",
    )
    args = parser.parse_args()

    output_path = run(args.config)
    print(f"Saved {output_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
