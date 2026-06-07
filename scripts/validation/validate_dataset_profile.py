"""Validate dataset profiling outputs."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    required = [
        "data/reports/mvtec_dataset_profile.csv",
        "data/reports/mvtec_image_metadata.csv",
        "data/reports/mvtec_dataset_profile_summary.json",
        "data/reports/mvtec_dataset_report.md",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]

    if missing:
        print("Dataset profiling validation failed:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("Dataset profiling validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
