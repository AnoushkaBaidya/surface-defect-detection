"""
Runtime model loading helpers.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from defect_detection.models.registry import build_registered_model


def load_classification_model(
    model_name: str,
    checkpoint_path: str | Path | None = None,
    *,
    pretrained: bool = False,
    device: str = "cpu",
) -> nn.Module:
    """Load a registered classification model and optional checkpoint."""
    model = build_registered_model(model_name, pretrained=pretrained)

    if checkpoint_path is not None:
        checkpoint = torch.load(Path(checkpoint_path), map_location=device)
        state_dict = (
            checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        )
        model.load_state_dict(state_dict)

    model.eval()
    return model.to(device)
