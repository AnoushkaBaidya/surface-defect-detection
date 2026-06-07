"""
Run the WinCLIP benchmark across all MVTec AD categories and k-shot settings.

Run all:
    python scripts/run_winclip_benchmark.py

Smoke test:
    python scripts/run_winclip_benchmark.py --categories bottle screw --k-shot-values 0 1
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
    """Run the WinCLIP benchmark across all MVTec AD categories and k-shot settings."""
    parser = argparse.ArgumentParser(description="Run the WinCLIP benchmark.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional category subset. Defaults to all configured categories.",
    )
    parser.add_argument(
        "--k-shot-values",
        nargs="+",
        type=int,
        default=None,
        help="Optional k-shot values. Defaults to configured values.",
    )
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/winclip_benchmark.yaml")

    categories = args.categories or config["dataset"]["categories"]
    k_shot_values = args.k_shot_values or config["model"]["k_shot_values"]

    logger.info("Starting WinCLIP benchmark WinCLIP benchmark.")
    logger.info("Categories: %s", categories)
    logger.info("k-shot values: %s", k_shot_values)

    for k_shot in k_shot_values:
        for category in categories:
            run_command(
                [
                    sys.executable,
                    "scripts/train_winclip.py",
                    "--category",
                    category,
                    "--k-shot",
                    str(k_shot),
                ]
            )

    logger.info("WinCLIP benchmark WinCLIP benchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
