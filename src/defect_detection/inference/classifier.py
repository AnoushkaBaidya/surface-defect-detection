"""
Thin service wrapper around runtime classification inference.
"""

from __future__ import annotations

from pathlib import Path

from defect_detection.runtime.model_loader import load_classification_model
from defect_detection.runtime.predictor import PredictionResult, predict_image


class ImageClassifierService:
    """Load a classification model and serve predictions."""

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: str | Path | None = None,
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.image_size = image_size
        self.model = load_classification_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            pretrained=pretrained,
            device=device,
        )

    def predict(self, image_path: str | Path) -> PredictionResult:
        """Predict one image path."""
        return predict_image(
            self.model,
            image_path,
            image_size=self.image_size,
            device=self.device,
            model_name=self.model_name,
        )
