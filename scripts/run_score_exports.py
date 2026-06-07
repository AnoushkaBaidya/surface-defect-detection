"""
Export scores for all hybrid source models.

Smoke test:
    python scripts/run_score_exports.py --categories bottle screw

All:
    python scripts/run_score_exports.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=None)
    args = parser.parse_args()

    categories = args.categories or CATEGORIES

    for category in categories:
        # run([sys.executable, "scripts/export_model_scores.py", "--model", "patchcore", "--category", category])
        # run([sys.executable, "scripts/export_model_scores.py", "--model", "fastflow", "--category", category])
        # run([sys.executable, "scripts/export_model_scores.py", "--model", "winclip", "--category", category, "--k-shot", "1"])
        run(
            [
                sys.executable,
                "scripts/export_model_scores.py",
                "--model",
                "winclip",
                "--category",
                category,
                "--k-shot",
                "4",
            ]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
