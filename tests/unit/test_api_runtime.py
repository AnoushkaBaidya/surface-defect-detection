from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image

if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None:
    pytestmark = pytest.mark.skip(reason="FastAPI runtime extras are not installed.")
else:
    from fastapi.testclient import TestClient

    from defect_detection.api.app import app
    from defect_detection.training.cnn_model import build_resnet18_binary_classifier


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint(tmp_path: Path) -> None:
    client = TestClient(app)
    checkpoint_path = tmp_path / "model.pt"
    model = build_resnet18_binary_classifier(pretrained=False)
    torch.save(model.state_dict(), checkpoint_path)

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(200, 200, 200)).save(image_path)

    response = client.post(
        "/v1/predict/classification",
        json={
            "image_path": str(image_path),
            "model_name": "resnet18_binary_classifier",
            "checkpoint_path": str(checkpoint_path),
            "pretrained": False,
            "device": "cpu",
            "image_size": 64,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "resnet18_binary_classifier"
    assert payload["predicted_label"] in {"normal", "defective"}
