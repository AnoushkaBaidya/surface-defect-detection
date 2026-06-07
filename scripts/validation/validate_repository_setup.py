"""Validate that the repository layout is intact."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REQUIRED_DIRECTORIES = [
    "configs",
    "configs/data",
    "configs/experiments",
    "docs",
    "reports",
    "scripts",
    "src/defect_detection",
    "tests",
]

REQUIRED_FILES = [
    "README.md",
    "PROJECT_OVERVIEW.md",
    "pyproject.toml",
    "requirements.txt",
    "configs/project.yaml",
    "configs/data/mvtec.yaml",
    "docs/limitations.md",
    "docs/reproducibility.md",
    "docs/runtime_tradeoffs.md",
    "docs/model_selection.md",
    "src/defect_detection/api/app.py",
    "src/defect_detection/runtime/predictor.py",
]


def main() -> int:
    from defect_detection.utils.config import load_yaml_config

    missing: list[str] = []

    for directory in REQUIRED_DIRECTORIES:
        if not (PROJECT_ROOT / directory).is_dir():
            missing.append(directory)

    for file_path in REQUIRED_FILES:
        if not (PROJECT_ROOT / file_path).is_file():
            missing.append(file_path)

    load_yaml_config(PROJECT_ROOT / "configs/project.yaml")
    load_yaml_config(PROJECT_ROOT / "configs/data/mvtec.yaml")

    if missing:
        print("Repository validation failed:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("Repository validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
