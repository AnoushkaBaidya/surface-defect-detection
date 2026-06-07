"""
Typed configuration models and validation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defect_detection.utils.config import load_yaml_config


@dataclass(frozen=True)
class ProjectMetadata:
    name: str
    version: str
    description: str


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    model_artifacts_dir: Path
    heatmap_artifacts_dir: Path
    benchmark_artifacts_dir: Path
    mlflow_tracking_uri: str


@dataclass(frozen=True)
class ProjectConfig:
    project: ProjectMetadata
    paths: ProjectPaths
    experiment_current: str
    experiment_name: str


@dataclass(frozen=True)
class DatasetDefinition:
    name: str
    display_name: str
    root_dir: Path
    archive_path: Path
    official_page: str
    archive_url: str
    primary_category: str
    optional_category: str


@dataclass(frozen=True)
class ProfilingConfig:
    image_extensions: tuple[str, ...]
    selected_categories: tuple[str, ...]
    sample_grid_max_images: int
    thumbnail_size: int


@dataclass(frozen=True)
class DatasetConfig:
    dataset: DatasetDefinition
    profiling: ProfilingConfig


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected '{key}' to be a non-empty string.")
    return value


def _require_number(payload: dict[str, Any], key: str) -> int | float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected '{key}' to be numeric.")
    return value


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected '{key}' to be a non-empty list.")
    return value


def load_project_config(config_path: str | Path) -> ProjectConfig:
    payload = load_yaml_config(config_path)

    project = _require_mapping(payload, "project")
    paths = _require_mapping(payload, "paths")
    experiment = _require_mapping(payload, "experiment")

    return ProjectConfig(
        project=ProjectMetadata(
            name=_require_str(project, "name"),
            version=_require_str(project, "version"),
            description=_require_str(project, "description"),
        ),
        paths=ProjectPaths(
            data_root=Path(_require_str(paths, "data_root")),
            raw_data_dir=Path(_require_str(paths, "raw_data_dir")),
            interim_data_dir=Path(_require_str(paths, "interim_data_dir")),
            processed_data_dir=Path(_require_str(paths, "processed_data_dir")),
            reports_dir=Path(_require_str(paths, "reports_dir")),
            artifacts_dir=Path(_require_str(paths, "artifacts_dir")),
            model_artifacts_dir=Path(_require_str(paths, "model_artifacts_dir")),
            heatmap_artifacts_dir=Path(_require_str(paths, "heatmap_artifacts_dir")),
            benchmark_artifacts_dir=Path(_require_str(paths, "benchmark_artifacts_dir")),
            mlflow_tracking_uri=_require_str(paths, "mlflow_tracking_uri"),
        ),
        experiment_current=_require_str(experiment, "current"),
        experiment_name=_require_str(experiment, "name"),
    )


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    payload = load_yaml_config(config_path)

    dataset = _require_mapping(payload, "dataset")
    profiling = _require_mapping(payload, "profiling")
    sample_grid = _require_mapping(profiling, "sample_grid")

    return DatasetConfig(
        dataset=DatasetDefinition(
            name=_require_str(dataset, "name"),
            display_name=_require_str(dataset, "display_name"),
            root_dir=Path(_require_str(dataset, "root_dir")),
            archive_path=Path(_require_str(dataset, "archive_path")),
            official_page=_require_str(dataset, "official_page"),
            archive_url=_require_str(dataset, "archive_url"),
            primary_category=_require_str(dataset, "primary_category"),
            optional_category=_require_str(dataset, "optional_category"),
        ),
        profiling=ProfilingConfig(
            image_extensions=tuple(
                str(item) for item in _require_list(profiling, "image_extensions")
            ),
            selected_categories=tuple(
                str(item) for item in _require_list(profiling, "selected_categories")
            ),
            sample_grid_max_images=int(_require_number(sample_grid, "max_images")),
            thumbnail_size=int(_require_number(sample_grid, "thumbnail_size")),
        ),
    )
