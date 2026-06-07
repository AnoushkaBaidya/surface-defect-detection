from __future__ import annotations

from pathlib import Path

import pytest

from defect_detection.utils.config import load_yaml_config
from defect_detection.utils.config_models import load_dataset_config, load_project_config


def test_load_yaml_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "missing.yaml")


def test_project_config_is_typed(project_root: Path) -> None:
    config = load_project_config(project_root / "configs/project.yaml")

    assert config.project.name == "surface-defect-detection"
    assert config.experiment_current == "mvtec_benchmark"


def test_dataset_config_is_typed(project_root: Path) -> None:
    config = load_dataset_config(project_root / "configs/data/mvtec.yaml")

    assert config.dataset.name == "mvtec_ad"
    assert config.profiling.thumbnail_size == 224
