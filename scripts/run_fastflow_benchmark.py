"""
Run the FastFlow benchmark across all MVTec AD categories.

Run:
    python scripts/run_fastflow_benchmark.py

Smoke test:
    python scripts/run_fastflow_benchmark.py --categories bottle screw
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
    """Run command and fail fast."""
    logger.info("Running command: %s", " ".join(command))

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def main() -> int:
    """Run FastFlow for all configured categories."""
    parser = argparse.ArgumentParser(description="Run the FastFlow benchmark.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional subset. Defaults to all configured categories.",
    )
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/fastflow_benchmark.yaml")

    categories = args.categories or config["dataset"]["categories"]

    logger.info("Starting FastFlow benchmark FastFlow benchmark.")
    logger.info("Categories: %s", categories)

    for category in categories:
        run_command(
            [
                sys.executable,
                "scripts/train_fastflow.py",
                "--category",
                category,
            ]
        )

    logger.info("FastFlow benchmark FastFlow benchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
