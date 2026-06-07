"""Validate typed project and dataset configurations."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    from defect_detection.utils.config_models import (
        load_dataset_config,
        load_project_config,
    )

    load_project_config(PROJECT_ROOT / "configs/project.yaml")
    load_dataset_config(PROJECT_ROOT / "configs/data/mvtec.yaml")

    experiment_dir = PROJECT_ROOT / "configs" / "experiments"
    yaml_files = sorted(experiment_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError("No experiment configuration files found.")

    for config_path in yaml_files:
        if config_path.name == ".gitkeep":
            continue
        payload = config_path.read_text(encoding="utf-8")
        if "experiment:" not in payload:
            raise ValueError(f"Missing experiment block in {config_path.name}")

    print("Configuration validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
