from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("surface_defect_api")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))

if not logger.handlers:
    logger.addHandler(handler)


def log_json(event: str, payload: dict[str, Any]) -> None:
    logger.info(
        json.dumps(
            {
                "event": event,
                **payload,
            },
            default=str,
        )
    )
