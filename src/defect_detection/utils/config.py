"""
Configuration utilities for the surface defect detection project.

Why this file exists
--------------------
This project avoids hardcoding paths and project settings
inside training, evaluation, or support scripts. This helper centralizes
YAML loading so benchmark scripts can read consistent project configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file into a Python dictionary.

    Parameters
    ----------
    config_path:
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML file is empty or invalid.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping/object: {path}")

    return config
