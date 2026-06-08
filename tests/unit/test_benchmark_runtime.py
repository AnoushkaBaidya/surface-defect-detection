from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from defect_detection.models.registry import build_registered_model
from defect_detection.optimization.runtime_optimization import (
    export_resnet18_to_onnx,
    fuse_scores,
    normalize_scores,
    write_json,
)

if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None:
    FASTAPI_AVAILABLE = False
else:
    FASTAPI_AVAILABLE = True
    from fastapi.testclient import TestClient

    from defect_detection.api.app import app


def test_score_fusion_math() -> None:
    anomaly_scores = [0.2, 0.5, 0.8]
    vlm_scores = [0.1, 0.6, 0.9]

    anomaly_norm = normalize_scores(anomaly_scores)
    vlm_norm = normalize_scores(vlm_scores)
    fused = fuse_scores(anomaly_scores, vlm_scores, anomaly_weight=0.7, vlm_weight=0.3)

    expected = (0.7 * anomaly_norm) + (0.3 * vlm_norm)
    assert pytest.approx(fused.tolist()) == expected.tolist()


def test_onnx_export_produces_file(tmp_path: Path) -> None:
    model = build_registered_model("resnet18_binary_classifier", pretrained=False)
    model.eval()

    output_path = tmp_path / "resnet18_test.onnx"
    exported_path = export_resnet18_to_onnx(model, output_path, image_size=64)

    assert exported_path.exists()
    assert exported_path.stat().st_size > 0


def test_quantization_json_has_required_keys(tmp_path: Path) -> None:
    output_path = tmp_path / "int8_quantization_results.json"
    payload = {
        "onnx_fp32": {"model_size_mb": 10.0, "mean_latency_ms": 5.0, "p95_latency_ms": 6.0, "f1": 0.9},
        "onnx_int8": {"model_size_mb": 5.0, "mean_latency_ms": 4.0, "p95_latency_ms": 5.0, "f1": 0.89},
        "deltas": {
            "size_delta_mb": 5.0,
            "size_reduction_pct": 50.0,
            "mean_latency_delta_ms": -1.0,
            "p95_latency_delta_ms": -1.0,
            "f1_delta": -0.01,
        },
    }

    write_json(payload, output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert "onnx_fp32" in loaded
    assert "onnx_int8" in loaded
    assert "deltas" in loaded
    assert {"size_delta_mb", "size_reduction_pct", "mean_latency_delta_ms", "p95_latency_delta_ms", "f1_delta"} <= set(
        loaded["deltas"]
    )


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI runtime extras are not installed.")
def test_resnet_fastapi_endpoint_regression(tmp_path: Path) -> None:
    client = TestClient(app)

    checkpoint_path = tmp_path / "model.pt"
    model = build_registered_model("resnet18_binary_classifier", pretrained=False)
    torch.save(model.state_dict(), checkpoint_path)

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(150, 150, 150)).save(image_path)

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
