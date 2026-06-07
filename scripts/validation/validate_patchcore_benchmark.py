"""Validate PatchCore benchmark outputs."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    required = [
        "artifacts/benchmarks/patchcore/per_category_results.csv",
        "artifacts/benchmarks/patchcore/aggregate_results.csv",
        "artifacts/benchmarks/patchcore/patchcore_benchmark_report.md",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]

    if missing:
        print("PatchCore benchmark validation failed:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("PatchCore benchmark validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
