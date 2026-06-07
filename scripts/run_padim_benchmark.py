"""
Run the PaDiM benchmark across all MVTec AD categories.

This script executes categories sequentially to keep failures isolated.

Run:
    python scripts/run_padim_benchmark.py

Optional smoke test:
    python scripts/run_padim_benchmark.py --categories bottle screw
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.utils.config import load_yaml_config  # noqa: E402
from defect_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def run_command(command: list[str]) -> None:
    """Run a command and fail fast on errors."""
    logger.info("Running command: %s", " ".join(command))

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def main() -> int:
    """Run all PaDiM category benchmarks."""
    parser = argparse.ArgumentParser(description="Run the PaDiM benchmark.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional subset. Defaults to all configured categories.",
    )
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/padim_benchmark.yaml")
    categories = args.categories or config["dataset"]["categories"]

    logger.info("Starting PaDiM benchmark PaDiM benchmark.")
    logger.info("Categories: %s", categories)

    for category in categories:
        run_command(
            [
                sys.executable,
                "scripts/train_padim.py",
                "--category",
                category,
            ]
        )

    logger.info("PaDiM benchmark PaDiM benchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
