"""
Build a benchmark summary across the evaluated models.

This script aggregates:
- ResNet18
- PaDiM
- PatchCore
- FastFlow

It computes:
- aggregate rankings
- runtime rankings
- quality rankings
- per-category comparisons
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from defect_detection.benchmark.summary import (
    compute_aggregate_metrics,
    compute_runtime_ranking,
    load_model_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks" / "summary"

RAW_OUTPUT_DIR = OUTPUT_DIR / "raw"

MODEL_SOURCES = {
    "resnet18": (PROJECT_ROOT / "artifacts" / "benchmarks" / "resnet18" / "raw_metrics"),
    "padim": (PROJECT_ROOT / "artifacts" / "benchmarks" / "padim" / "raw_metrics"),
    "patchcore": (PROJECT_ROOT / "artifacts" / "benchmarks" / "patchcore" / "raw_metrics"),
    "fastflow": (PROJECT_ROOT / "artifacts" / "benchmarks" / "fastflow" / "raw_metrics"),
}


def main() -> None:
    """
    Build unified benchmark outputs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_model_frames = []

    for model_name, metrics_dir in MODEL_SOURCES.items():
        print(f"Loading metrics for: {model_name}")

        model_df = load_model_metrics(
            model_name=model_name,
            metrics_dir=metrics_dir,
        )

        if not model_df.empty:
            all_model_frames.append(model_df)

            raw_output_path = RAW_OUTPUT_DIR / f"{model_name}_all_categories.csv"

            model_df.to_csv(raw_output_path, index=False)

    combined_df = pd.concat(
        all_model_frames,
        ignore_index=True,
    )

    aggregate_df = compute_aggregate_metrics(combined_df)

    runtime_ranking_df = compute_runtime_ranking(
        aggregate_df,
    )

    combined_output_path = OUTPUT_DIR / "all_models_per_category.csv"

    aggregate_output_path = OUTPUT_DIR / "aggregate_model_comparison.csv"

    runtime_output_path = OUTPUT_DIR / "runtime_ranking.csv"

    combined_df.to_csv(combined_output_path, index=False)

    aggregate_df.to_csv(aggregate_output_path, index=False)

    runtime_ranking_df.to_csv(
        runtime_output_path,
        index=False,
    )

    print("\nUnified benchmark generation complete.")
    print(f"Saved: {combined_output_path}")
    print(f"Saved: {aggregate_output_path}")
    print(f"Saved: {runtime_output_path}")


if __name__ == "__main__":
    main()
