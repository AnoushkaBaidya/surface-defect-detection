"""
Image classification runtime predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class PredictionResult:
    model_name: str
    predicted_label: str
    predicted_index: int
    defect_probability: float
    normal_probability: float
    image_path: str


def build_classification_transform(image_size: int) -> transforms.Compose:
    """Build deterministic runtime transform."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    *,
    image_size: int = 224,
    device: str = "cpu",
    model_name: str = "unknown",
) -> PredictionResult:
    """Run classification inference for one image path."""
    image = Image.open(image_path).convert("RGB")
    transform = build_classification_transform(image_size)
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0].cpu()

    normal_probability = float(probabilities[0].item())
    defect_probability = float(probabilities[1].item())
    predicted_index = int(torch.argmax(probabilities).item())
    predicted_label = "defective" if predicted_index == 1 else "normal"

    return PredictionResult(
        model_name=model_name,
        predicted_label=predicted_label,
        predicted_index=predicted_index,
        defect_probability=defect_probability,
        normal_probability=normal_probability,
        image_path=str(image_path),
    )
