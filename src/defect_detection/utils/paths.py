"""
Path utilities for project-wide filesystem handling.

Why this file exists
--------------------
ML projects often fail because scripts assume they are executed from one
specific folder. These helpers make path handling more reliable and readable.
"""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """
    Return the project root directory.

    This assumes the file lives under:
    src/defect_detection/utils/paths.py

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[3]


def resolve_project_path(relative_path: str | Path) -> Path:
    """
    Resolve a path relative to the project root.

    Parameters
    ----------
    relative_path:
        Path relative to the repository root.

    Returns
    -------
    Path
        Absolute resolved path.
    """
    return get_project_root() / Path(relative_path)
