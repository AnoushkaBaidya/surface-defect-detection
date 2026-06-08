"""
Quantize the exported bottle-category ResNet-18 ONNX model and benchmark INT8.
"""

from __future__ import annotations

import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.optimization.runtime_optimization import (  # noqa: E402
    benchmark_onnx_classifier,
    build_onnx_session,
    model_size_mb,
    read_bottle_eval_split,
    write_json,
)

FP32_JSON = PROJECT_ROOT / "benchmark_outputs/onnx_fp32_results.json"
FP32_ONNX = PROJECT_ROOT / "benchmark_outputs/models/resnet18_bottle_fp32.onnx"
INT8_ONNX = PROJECT_ROOT / "benchmark_outputs/models/resnet18_bottle_int8.onnx"
JSON_OUTPUT = PROJECT_ROOT / "benchmark_outputs/int8_quantization_results.json"


def main() -> int:
    """Quantize the ONNX model and benchmark FP32 vs INT8."""
    if not FP32_JSON.exists():
        raise FileNotFoundError("Run scripts/export_and_benchmark_onnx.py first.")
    if not FP32_ONNX.exists():
        raise FileNotFoundError(FP32_ONNX)

    eval_df = read_bottle_eval_split(PROJECT_ROOT)
    image_paths = [PROJECT_ROOT / image_path for image_path in eval_df["image_path"].tolist()]
    labels = [1 if label == "defective" else 0 for label in eval_df["label"].tolist()]

    quantize_dynamic(
        str(FP32_ONNX),
        str(INT8_ONNX),
        weight_type=QuantType.QInt8,
    )

    fp32_session = build_onnx_session(FP32_ONNX)
    int8_session = build_onnx_session(INT8_ONNX)

    fp32_results = benchmark_onnx_classifier(fp32_session, image_paths, labels, image_size=224)
    int8_results = benchmark_onnx_classifier(int8_session, image_paths, labels, image_size=224)

    fp32_size = model_size_mb(FP32_ONNX)
    int8_size = model_size_mb(INT8_ONNX)
    size_delta = fp32_size - int8_size
    size_reduction_pct = (size_delta / fp32_size) * 100.0 if fp32_size > 0 else 0.0

    payload = {
        "artifact_sources": {
            "fp32_onnx": str(FP32_ONNX.relative_to(PROJECT_ROOT)),
            "int8_onnx": str(INT8_ONNX.relative_to(PROJECT_ROOT)),
            "bottle_eval_split": "data/processed/phase3/resnet18/splits/bottle/eval.csv",
        },
        "hybrid_note": (
            "The hybrid winner is a component-based serving policy. PatchCore and WinCLIP are not "
            "quantized in this pass; the INT8 benchmark covers the deployable ResNet-18 baseline only."
        ),
        "onnx_fp32": {
            "model_size_mb": fp32_size,
            **fp32_results["latency"],
            **fp32_results["classification"],
        },
        "onnx_int8": {
            "model_size_mb": int8_size,
            **int8_results["latency"],
            **int8_results["classification"],
        },
        "deltas": {
            "size_delta_mb": size_delta,
            "size_reduction_pct": size_reduction_pct,
            "mean_latency_delta_ms": float(
                int8_results["latency"]["mean_latency_ms"] - fp32_results["latency"]["mean_latency_ms"]
            ),
            "p95_latency_delta_ms": float(
                int8_results["latency"]["p95_latency_ms"] - fp32_results["latency"]["p95_latency_ms"]
            ),
            "f1_delta": float(
                int8_results["classification"]["f1"] - fp32_results["classification"]["f1"]
            ),
        },
    }

    write_json(payload, JSON_OUTPUT)
    print(JSON_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
