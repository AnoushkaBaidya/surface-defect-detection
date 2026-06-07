from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from defect_detection.inference.classifier import ImageClassifierService
from defect_detection.training.cnn_model import build_resnet18_binary_classifier


def test_image_classifier_service_runs_prediction(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    model = build_resnet18_binary_classifier(pretrained=False)
    torch.save(model.state_dict(), checkpoint_path)

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 10, 10)).save(image_path)

    service = ImageClassifierService(
        model_name="resnet18_binary_classifier",
        checkpoint_path=checkpoint_path,
        pretrained=False,
        device="cpu",
        image_size=32,
    )
    result = service.predict(image_path)

    assert result.model_name == "resnet18_binary_classifier"
    assert result.predicted_label in {"normal", "defective"}
    assert 0.0 <= result.defect_probability <= 1.0
