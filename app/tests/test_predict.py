from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import app.main as app_main
from app.config import (
    ApiLimitsConfig,
    AppConfig,
    EfficientADConfig,
    ModelPolicyConfig,
    ResNetConfig,
)


class FakeResNetService:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict_score(self, image_tensor) -> float:  # noqa: ANN001
        return self.score


class FakeEfficientADService:
    def __init__(self, raw_score: float, normalized_score: float) -> None:
        self.raw_score = raw_score
        self.normalized_score = normalized_score

    def predict_scores(self, image_tensor) -> dict[str, float]:  # noqa: ANN001
        return {
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
        }


def make_config() -> AppConfig:
    return AppConfig(
        model_policy=ModelPolicyConfig(name="test_policy", version="test"),
        resnet18=ResNetConfig(
            onnx_path="models/resnet18_internal.onnx",
            input_size=224,
            raw_threshold=0.5,
        ),
        efficientad=EfficientADConfig(
            enabled=True,
            threshold=0.3,
            runtime="onnx",
            onnx_path="models/efficientad_score.onnx",
            calibration_path="configs/efficientad_calibration.json",
            input_size=224,
        ),
        api_limits=ApiLimitsConfig(
            max_upload_bytes=1024 * 1024,
            min_image_dimension=32,
            max_image_dimension=2048,
        ),
    )


def make_image_bytes(size: tuple[int, int] = (128, 128)) -> bytes:
    image = Image.new("RGB", size=size, color=(120, 120, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_resnet_positive(monkeypatch) -> None:
    monkeypatch.setattr(app_main, "load_models", lambda: None)
    monkeypatch.setattr(app_main, "config", make_config())
    monkeypatch.setattr(app_main, "resnet_service", FakeResNetService(score=0.9))
    monkeypatch.setattr(app_main, "efficientad_service", FakeEfficientADService(0.1, 0.1))

    with TestClient(app_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", make_image_bytes(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "defective"
    assert payload["decision_reason"] == "resnet18_score_above_threshold"


def test_predict_uses_efficientad_gate(monkeypatch) -> None:
    monkeypatch.setattr(app_main, "load_models", lambda: None)
    monkeypatch.setattr(app_main, "config", make_config())
    monkeypatch.setattr(app_main, "resnet_service", FakeResNetService(score=0.1))
    monkeypatch.setattr(app_main, "efficientad_service", FakeEfficientADService(0.7, 0.4))

    with TestClient(app_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", make_image_bytes(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "defective"
    assert payload["decision_reason"] == "efficientad_high_confidence_gate"


def test_predict_rejects_large_payload(monkeypatch) -> None:
    monkeypatch.setattr(app_main, "load_models", lambda: None)
    config = make_config()
    config.api_limits.max_upload_bytes = 10
    monkeypatch.setattr(app_main, "config", config)
    monkeypatch.setattr(app_main, "resnet_service", FakeResNetService(score=0.1))
    monkeypatch.setattr(app_main, "efficientad_service", FakeEfficientADService(0.1, 0.1))

    with TestClient(app_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", make_image_bytes(), "image/png")},
        )

    assert response.status_code == 413


def test_predict_rejects_small_image(monkeypatch) -> None:
    monkeypatch.setattr(app_main, "load_models", lambda: None)
    monkeypatch.setattr(app_main, "config", make_config())
    monkeypatch.setattr(app_main, "resnet_service", FakeResNetService(score=0.1))
    monkeypatch.setattr(app_main, "efficientad_service", FakeEfficientADService(0.1, 0.1))

    with TestClient(app_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("tiny.png", make_image_bytes((8, 8)), "image/png")},
        )

    assert response.status_code == 400
    assert "too small" in response.json()["detail"]


def test_predict_handles_runtime_failure(monkeypatch) -> None:
    class ExplodingService:
        def predict_score(self, image_tensor) -> float:  # noqa: ANN001
            raise RuntimeError("boom")

    monkeypatch.setattr(app_main, "load_models", lambda: None)
    monkeypatch.setattr(app_main, "config", make_config())
    monkeypatch.setattr(app_main, "resnet_service", ExplodingService())
    monkeypatch.setattr(app_main, "efficientad_service", None)

    with TestClient(app_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", make_image_bytes(), "image/png")},
        )

    assert response.status_code == 500
