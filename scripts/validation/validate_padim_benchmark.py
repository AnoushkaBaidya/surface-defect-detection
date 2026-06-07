"""Validate PaDiM benchmark outputs."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    required = [
        "artifacts/benchmarks/padim/per_category_results.csv",
        "artifacts/benchmarks/padim/aggregate_results.csv",
        "artifacts/benchmarks/padim/padim_benchmark_report.md",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]

    if missing:
        print("PaDiM benchmark validation failed:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("PaDiM benchmark validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
