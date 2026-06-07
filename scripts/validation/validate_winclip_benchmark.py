"""Validate WinCLIP benchmark outputs."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    required = [
        "artifacts/benchmarks/winclip/per_category_results.csv",
        "artifacts/benchmarks/winclip/aggregate_results.csv",
        "artifacts/benchmarks/winclip/winclip_benchmark_report.md",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]

    if missing:
        print("WinCLIP benchmark validation failed:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("WinCLIP benchmark validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
