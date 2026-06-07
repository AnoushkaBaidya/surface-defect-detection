"""
Shared helpers for training and evaluation scripts.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import torch


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configure MLflow tracking and experiment selection."""
    if tracking_uri.startswith(("http://", "https://", "sqlite://", "file:")):
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(Path(tracking_uri).resolve().as_uri())

    mlflow.set_experiment(experiment_name)


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_directory_size_mb(directory: Path) -> float:
    """Calculate recursive directory size in megabytes."""
    if not directory.exists():
        return 0.0

    total_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    return total_bytes / (1024 * 1024)
