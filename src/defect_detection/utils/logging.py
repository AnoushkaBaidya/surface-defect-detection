"""
Logging utilities for consistent project logging.

Why this file exists
--------------------
Production ML code should not rely only on print statements. A consistent
logger makes debugging easier across scripts, training jobs, APIs, and batch
pipelines.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Parameters
    ----------
    name:
        Logger name, usually __name__ from the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
