"""Validate key package imports and runtime module wiring."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _import_or_skip(module_name: str, *, dependency: str | None = None) -> None:
    """Import a module, skipping optional surfaces when extras are unavailable."""
    if dependency is not None and importlib.util.find_spec(dependency) is None:
        print(f"Skipping optional import: {module_name} (missing dependency: {dependency})")
        return
    importlib.import_module(module_name)


def main() -> int:
    _import_or_skip("defect_detection.benchmark.summary")
    _import_or_skip("defect_detection.benchmark.splits")
    _import_or_skip("defect_detection.monitoring.drift")
    _import_or_skip("defect_detection.runtime.model_loader")
    _import_or_skip("defect_detection.runtime.predictor")
    _import_or_skip("defect_detection.utils.config_models")
    _import_or_skip("defect_detection.inference.classifier", dependency="torch")
    _import_or_skip("defect_detection.api.app", dependency="fastapi")

    print("Import validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
