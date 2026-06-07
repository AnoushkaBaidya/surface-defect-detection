"""
CNN model factory for the supervised baseline.

Why this file exists
--------------------
The CNN baseline is not the final model. It is a supervised reference
point used to show the limitations of standard classification under limited
defect labels.

We use ResNet18 because:
- it is simple
- it is widely recognized
- it trains quickly
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18_binary_classifier(pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet18 binary classifier.

    Parameters
    ----------
    pretrained:
        Whether to start from ImageNet weights.

    Returns
    -------
    nn.Module
        ResNet18 model with a 2-class output head.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 2)

    return model
