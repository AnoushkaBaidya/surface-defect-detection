"""
Run the ResNet18 benchmark across all configured categories.

Purpose
-------
This script executes category-specific ResNet18 training for every MVTec AD
category listed in the ResNet18 benchmark config.

It runs categories sequentially to keep logs readable and avoid overwhelming
a local machine.

Run:
    python scripts/run_resnet18_benchmark.py

Optional:
    python scripts/run_resnet18_benchmark.py --categories bottle screw
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
    """
    Run a subprocess command and fail fast if it errors.

    Parameters
    ----------
    command:
        Command tokens to execute.
    """
    logger.info("Running command: %s", " ".join(command))

    completed_process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed_process.returncode}: {command}"
        )


def main() -> int:
    """Run all configured category-specific ResNet18 benchmarks."""
    parser = argparse.ArgumentParser(description="Run the ResNet18 benchmark.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional subset of categories. Defaults to all configured categories.",
    )
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/resnet18_benchmark.yaml")

    categories = args.categories or config["dataset"]["categories"]

    logger.info("Starting ResNet18 benchmark ResNet18 benchmark.")
    logger.info("Categories: %s", categories)

    for category in categories:
        logger.info("Starting category: %s", category)

        run_command(
            [
                sys.executable,
                "scripts/train_resnet18.py",
                "--category",
                category,
            ]
        )

        logger.info("Completed category: %s", category)

    logger.info("ResNet18 benchmark ResNet18 benchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
