"""
Simple model registry for runtime loading.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from defect_detection.training.cnn_model import build_resnet18_binary_classifier

ModelFactory = Callable[..., nn.Module]


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "resnet18_binary_classifier": build_resnet18_binary_classifier,
}


def get_registered_model_names() -> tuple[str, ...]:
    """Return sorted supported runtime model names."""
    return tuple(sorted(MODEL_REGISTRY))


def build_registered_model(model_name: str, *, pretrained: bool = False) -> nn.Module:
    """Instantiate a supported model by name."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unsupported model_name '{model_name}'.")

    return MODEL_REGISTRY[model_name](pretrained=pretrained)
