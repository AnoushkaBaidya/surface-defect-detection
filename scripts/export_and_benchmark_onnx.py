"""
Export the public bottle-category ResNet-18 checkpoint to ONNX and benchmark it.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.models.registry import build_registered_model  # noqa: E402
from defect_detection.optimization.runtime_optimization import (  # noqa: E402
    benchmark_onnx_classifier,
    benchmark_pytorch_classifier,
    build_onnx_session,
    export_resnet18_to_onnx,
    load_resnet_checkpoint,
    max_probability_delta,
    mean_probability_delta,
    model_size_mb,
    read_bottle_eval_split,
    write_json,
)

RESNET_CHECKPOINT = PROJECT_ROOT / "mlruns/526830661020181409/1e40f7369746481cac631eada887a24c/artifacts/best_model.pt"
ONNX_OUTPUT = PROJECT_ROOT / "benchmark_outputs/models/resnet18_bottle_fp32.onnx"
JSON_OUTPUT = PROJECT_ROOT / "benchmark_outputs/onnx_fp32_results.json"


def main() -> int:
    """Export ResNet-18 to ONNX and benchmark PyTorch vs ONNX FP32."""
    if not RESNET_CHECKPOINT.exists():
        raise FileNotFoundError(RESNET_CHECKPOINT)

    eval_df = read_bottle_eval_split(PROJECT_ROOT)
    image_paths = [PROJECT_ROOT / image_path for image_path in eval_df["image_path"].tolist()]
    labels = [1 if label == "defective" else 0 for label in eval_df["label"].tolist()]

    checkpoint = load_resnet_checkpoint(RESNET_CHECKPOINT)
    model = build_registered_model("resnet18_binary_classifier", pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pytorch_results = benchmark_pytorch_classifier(
        model,
        image_paths,
        labels,
        image_size=int(checkpoint.get("image_size", 224)),
        device="cpu",
    )

    export_resnet18_to_onnx(
        model,
        ONNX_OUTPUT,
        image_size=int(checkpoint.get("image_size", 224)),
    )
    session = build_onnx_session(ONNX_OUTPUT)

    onnx_results = benchmark_onnx_classifier(
        session,
        image_paths,
        labels,
        image_size=int(checkpoint.get("image_size", 224)),
    )

    payload = {
        "artifact_sources": {
            "resnet_checkpoint": str(RESNET_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "bottle_eval_split": "data/processed/phase3/resnet18/splits/bottle/eval.csv",
        },
        "exported_onnx_path": str(ONNX_OUTPUT.relative_to(PROJECT_ROOT)),
        "pytorch_fp32": {
            "model_size_mb": model_size_mb(RESNET_CHECKPOINT),
            **pytorch_results["latency"],
            **pytorch_results["classification"],
        },
        "onnx_fp32": {
            "model_size_mb": model_size_mb(ONNX_OUTPUT),
            **onnx_results["latency"],
            **onnx_results["classification"],
        },
        "parity": {
            "mean_defect_probability_delta": mean_probability_delta(
                pytorch_results["defect_scores"],
                onnx_results["defect_scores"],
            ),
            "max_defect_probability_delta": max_probability_delta(
                pytorch_results["defect_scores"],
                onnx_results["defect_scores"],
            ),
        },
        "deltas": {
            "f1_delta": float(
                onnx_results["classification"]["f1"] - pytorch_results["classification"]["f1"]
            ),
            "mean_latency_delta_ms": float(
                onnx_results["latency"]["mean_latency_ms"]
                - pytorch_results["latency"]["mean_latency_ms"]
            ),
            "p95_latency_delta_ms": float(
                onnx_results["latency"]["p95_latency_ms"]
                - pytorch_results["latency"]["p95_latency_ms"]
            ),
        },
    }

    write_json(payload, JSON_OUTPUT)
    print(JSON_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
