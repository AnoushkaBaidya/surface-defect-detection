"""
Benchmark bottle-category runtime latency from the public MLflow artifacts.

Hardware target
---------------
CPU_LOCAL_TARGET = "Intel64 Family 6 Model 186 Stepping 3, GenuineIntel"

This script materializes the runtime comparison using the tracked bottle-category
artifacts in `mlruns/`. It loads the public artifact paths for:
- ResNet-18 baseline checkpoint
- PatchCore bottle checkpoint
- WinCLIP k=4 bottle config

The latency and quality snapshots are sourced from the real benchmark artifacts:
- bottle-category per-model metric JSONs for baseline and components
- public hybrid comparison CSV for the winning hybrid quality snapshot
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.optimization.runtime_optimization import load_json, write_json  # noqa: E402

CPU_LOCAL_TARGET = "Intel64 Family 6 Model 186 Stepping 3, GenuineIntel"
BATCH_SIZE = 1

RESNET_CHECKPOINT = PROJECT_ROOT / "mlruns/526830661020181409/1e40f7369746481cac631eada887a24c/artifacts/best_model.pt"
RESNET_METRICS = PROJECT_ROOT / "mlruns/526830661020181409/1e40f7369746481cac631eada887a24c/artifacts/bottle_resnet18_metrics.json"
PATCHCORE_CHECKPOINT = PROJECT_ROOT / "mlruns/844624909902585835/dc7ccf7659d446889f0fbbdf483d5d50/artifacts/patchcore_model.ckpt"
PATCHCORE_METRICS = PROJECT_ROOT / "mlruns/844624909902585835/dc7ccf7659d446889f0fbbdf483d5d50/artifacts/bottle_patchcore_metrics.json"
WINCLIP_CONFIG = PROJECT_ROOT / "mlruns/676412444936204945/9c65793fafb747b58cff629a4ed6b104/artifacts/bottle_k4_winclip_config.json"
WINCLIP_METRICS = PROJECT_ROOT / "mlruns/676412444936204945/9c65793fafb747b58cff629a4ed6b104/artifacts/bottle_winclip_k4_metrics.json"
HYBRID_COMPARISON_CSV = PROJECT_ROOT / "reports/mlflow_model_comparison.csv"
OUTPUT_PATH = PROJECT_ROOT / "benchmark_outputs/runtime_latency_results.json"


def main() -> int:
    """Create the runtime latency comparison JSON."""
    required_paths = [
        RESNET_CHECKPOINT,
        RESNET_METRICS,
        PATCHCORE_CHECKPOINT,
        PATCHCORE_METRICS,
        WINCLIP_CONFIG,
        WINCLIP_METRICS,
        HYBRID_COMPARISON_CSV,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing runtime benchmark inputs: {missing}")

    resnet_metrics = load_json(RESNET_METRICS)
    patchcore_metrics = load_json(PATCHCORE_METRICS)
    winclip_metrics = load_json(WINCLIP_METRICS)
    hybrid_comparison = pd.read_csv(HYBRID_COMPARISON_CSV)

    hybrid_row = hybrid_comparison[
        hybrid_comparison["model_name"] == "patchcore_winclip_k4_aw0.7_vw0.3"
    ]
    if hybrid_row.empty:
        raise ValueError("Hybrid comparison row not found in reports/mlflow_model_comparison.csv")
    hybrid_row = hybrid_row.iloc[0]

    hybrid_mean_latency_ms = float(
        patchcore_metrics["inference_latency_ms_mean"] + winclip_metrics["inference_latency_ms_mean"]
    )
    hybrid_p95_latency_ms = float(
        patchcore_metrics["inference_latency_ms_p95"] + winclip_metrics["inference_latency_ms_p95"]
    )
    hybrid_throughput = float(1000.0 / hybrid_mean_latency_ms) if hybrid_mean_latency_ms > 0 else 0.0
    hybrid_artifact_size_mb = float(
        patchcore_metrics["model_size_mb"] + winclip_metrics["model_size_mb"]
    )

    payload = {
        "hardware": {
            "cpu_target": CPU_LOCAL_TARGET,
            "batch_size": BATCH_SIZE,
            "dataset_scope": "MVTec AD bottle held-out benchmark artifacts",
            "measurement_mode": "artifact-backed bottle-category runtime summary",
        },
        "baseline_resnet18": {
            "artifact_sources": {
                "checkpoint": str(RESNET_CHECKPOINT.relative_to(PROJECT_ROOT)),
                "metrics_json": str(RESNET_METRICS.relative_to(PROJECT_ROOT)),
            },
            "mean_latency_ms": float(resnet_metrics["inference_latency_ms_mean"]),
            "p95_latency_ms": float(resnet_metrics["inference_latency_ms_p95"]),
            "throughput_images_per_second": float(resnet_metrics["throughput_images_per_second"]),
            "f1": float(resnet_metrics["classification_f1"]),
            "recall": float(resnet_metrics["recall"]),
            "false_negative_rate": float(resnet_metrics["false_negative_rate"]),
            "model_size_mb": float(resnet_metrics["model_size_mb"]),
        },
        "patchcore_component": {
            "artifact_sources": {
                "checkpoint": str(PATCHCORE_CHECKPOINT.relative_to(PROJECT_ROOT)),
                "metrics_json": str(PATCHCORE_METRICS.relative_to(PROJECT_ROOT)),
            },
            "mean_latency_ms": float(patchcore_metrics["inference_latency_ms_mean"]),
            "p95_latency_ms": float(patchcore_metrics["inference_latency_ms_p95"]),
            "throughput_images_per_second": float(patchcore_metrics["throughput_images_per_second"]),
            "f1": float(patchcore_metrics["classification_f1"]),
            "recall": float(patchcore_metrics["recall"]),
            "false_negative_rate": float(patchcore_metrics["false_negative_rate"]),
            "model_size_mb": float(patchcore_metrics["model_size_mb"]),
        },
        "winclip_k4_component": {
            "artifact_sources": {
                "config": str(WINCLIP_CONFIG.relative_to(PROJECT_ROOT)),
                "metrics_json": str(WINCLIP_METRICS.relative_to(PROJECT_ROOT)),
            },
            "mean_latency_ms": float(winclip_metrics["inference_latency_ms_mean"]),
            "p95_latency_ms": float(winclip_metrics["inference_latency_ms_p95"]),
            "throughput_images_per_second": float(winclip_metrics["throughput_images_per_second"]),
            "f1": float(winclip_metrics["classification_f1"]),
            "recall": float(winclip_metrics["recall"]),
            "false_negative_rate": float(winclip_metrics["false_negative_rate"]),
            "model_size_mb": float(winclip_metrics["model_size_mb"]),
        },
        "hybrid_policy_patchcore_winclip_k4": {
            "artifact_sources": {
                "patchcore_checkpoint": str(PATCHCORE_CHECKPOINT.relative_to(PROJECT_ROOT)),
                "winclip_config": str(WINCLIP_CONFIG.relative_to(PROJECT_ROOT)),
                "quality_snapshot_csv": str(HYBRID_COMPARISON_CSV.relative_to(PROJECT_ROOT)),
            },
            "policy_description": "PatchCore score + WinCLIP k=4 score, min-max normalized and fused at 0.7 / 0.3",
            "mean_latency_ms": hybrid_mean_latency_ms,
            "p95_latency_ms": hybrid_p95_latency_ms,
            "throughput_images_per_second": hybrid_throughput,
            "f1": float(hybrid_row["classification_f1"]),
            "recall": float(hybrid_row["recall"]),
            "false_negative_rate": float(hybrid_row["false_negative_rate"]),
            "model_size_mb": hybrid_artifact_size_mb,
            "quality_snapshot_scope": "public final-selection aggregate snapshot",
            "latency_scope": "bottle component latencies summed sequentially",
        },
    }

    write_json(payload, OUTPUT_PATH)
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
