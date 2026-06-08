"""
Create the runtime optimization markdown report from benchmark JSON outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.optimization.runtime_optimization import load_json  # noqa: E402
from defect_detection.reporting.io import write_markdown  # noqa: E402

RUNTIME_JSON = PROJECT_ROOT / "benchmark_outputs/runtime_latency_results.json"
ONNX_JSON = PROJECT_ROOT / "benchmark_outputs/onnx_fp32_results.json"
INT8_JSON = PROJECT_ROOT / "benchmark_outputs/int8_quantization_results.json"
REPORT_PATH = PROJECT_ROOT / "reports/runtime_optimization_report.md"


def fmt(value: float, digits: int = 2) -> str:
    """Format a float consistently."""
    return f"{value:.{digits}f}"


def comparison_row(
    name: str,
    *,
    size_mb: float | None = None,
    mean_latency_ms: float,
    p95_latency_ms: float,
    throughput: float | None = None,
    f1: float,
    recall: float,
    false_negative_rate: float,
) -> str:
    """Build a compact markdown table row."""
    size_value = fmt(size_mb) if size_mb is not None else "N/A"
    throughput_value = fmt(throughput) if throughput is not None else "N/A"
    return (
        f"| {name} | {size_value} | {fmt(mean_latency_ms)} | "
        f"{fmt(p95_latency_ms)} | {throughput_value} | {fmt(f1, 4)} | "
        f"{fmt(recall, 4)} | {fmt(false_negative_rate, 4)} |"
    )


def main() -> int:
    """Create the runtime optimization report."""
    runtime = load_json(RUNTIME_JSON)
    onnx_fp32 = load_json(ONNX_JSON)
    int8 = load_json(INT8_JSON)

    baseline = runtime["baseline_resnet18"]
    hybrid = runtime["hybrid_policy_patchcore_winclip_k4"]
    onnx_baseline = onnx_fp32["onnx_fp32"]
    fp32_baseline = onnx_fp32["pytorch_fp32"]
    int8_baseline = int8["onnx_int8"]

    report = [
        "# Runtime Optimization Report",
        "",
        "## Hardware Context",
        "",
        f"- CPU target: `{runtime['hardware']['cpu_target']}`",
        f"- Batch size: `{runtime['hardware']['batch_size']}`",
        f"- Scope: {runtime['hardware']['dataset_scope']}",
        "",
        "## Raw Latency Comparison",
        "",
        "| Path | Size (MB) | Mean Latency (ms) | P95 Latency (ms) | Throughput (img/s) | F1 | Recall | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        comparison_row(
            "ResNet-18 baseline",
            size_mb=baseline["model_size_mb"],
            mean_latency_ms=baseline["mean_latency_ms"],
            p95_latency_ms=baseline["p95_latency_ms"],
            throughput=baseline["throughput_images_per_second"],
            f1=baseline["f1"],
            recall=baseline["recall"],
            false_negative_rate=baseline["false_negative_rate"],
        ),
        comparison_row(
            "PatchCore + WinCLIP k=4 hybrid",
            size_mb=hybrid["model_size_mb"],
            mean_latency_ms=hybrid["mean_latency_ms"],
            p95_latency_ms=hybrid["p95_latency_ms"],
            throughput=hybrid["throughput_images_per_second"],
            f1=hybrid["f1"],
            recall=hybrid["recall"],
            false_negative_rate=hybrid["false_negative_rate"],
        ),
        "",
        (
            "The hybrid latency is reported as a sequential component policy "
            "(`PatchCore` + `WinCLIP k=4`) using the bottle-category artifact timings. "
            "The quality snapshot comes from the public final-selection comparison CSV."
        ),
        "",
        "## ONNX FP32 Export Comparison",
        "",
        "| Path | Size (MB) | Mean Latency (ms) | P95 Latency (ms) | Throughput (img/s) | F1 | Recall | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        comparison_row(
            "PyTorch FP32",
            size_mb=fp32_baseline["model_size_mb"],
            mean_latency_ms=fp32_baseline["mean_latency_ms"],
            p95_latency_ms=fp32_baseline["p95_latency_ms"],
            throughput=fp32_baseline["throughput_images_per_second"],
            f1=fp32_baseline["f1"],
            recall=fp32_baseline["recall"],
            false_negative_rate=fp32_baseline["false_negative_rate"],
        ),
        comparison_row(
            "ONNX FP32",
            size_mb=onnx_baseline["model_size_mb"],
            mean_latency_ms=onnx_baseline["mean_latency_ms"],
            p95_latency_ms=onnx_baseline["p95_latency_ms"],
            throughput=onnx_baseline["throughput_images_per_second"],
            f1=onnx_baseline["f1"],
            recall=onnx_baseline["recall"],
            false_negative_rate=onnx_baseline["false_negative_rate"],
        ),
        "",
        f"- Mean defect-probability parity delta: {fmt(onnx_fp32['parity']['mean_defect_probability_delta'], 6)}",
        f"- Max defect-probability parity delta: {fmt(onnx_fp32['parity']['max_defect_probability_delta'], 6)}",
        "",
        "## INT8 Quantization Comparison",
        "",
        "| Path | Size (MB) | Mean Latency (ms) | P95 Latency (ms) | Throughput (img/s) | F1 | Recall | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        comparison_row(
            "ONNX FP32",
            size_mb=int8["onnx_fp32"]["model_size_mb"],
            mean_latency_ms=int8["onnx_fp32"]["mean_latency_ms"],
            p95_latency_ms=int8["onnx_fp32"]["p95_latency_ms"],
            throughput=int8["onnx_fp32"]["throughput_images_per_second"],
            f1=int8["onnx_fp32"]["f1"],
            recall=int8["onnx_fp32"]["recall"],
            false_negative_rate=int8["onnx_fp32"]["false_negative_rate"],
        ),
        comparison_row(
            "ONNX INT8",
            size_mb=int8_baseline["model_size_mb"],
            mean_latency_ms=int8_baseline["mean_latency_ms"],
            p95_latency_ms=int8_baseline["p95_latency_ms"],
            throughput=int8_baseline["throughput_images_per_second"],
            f1=int8_baseline["f1"],
            recall=int8_baseline["recall"],
            false_negative_rate=int8_baseline["false_negative_rate"],
        ),
        "",
        f"- Size reduction: {fmt(int8['deltas']['size_delta_mb'])} MB ({fmt(int8['deltas']['size_reduction_pct'])}%)",
        f"- Mean latency delta: {fmt(int8['deltas']['mean_latency_delta_ms'])} ms",
        f"- P95 latency delta: {fmt(int8['deltas']['p95_latency_delta_ms'])} ms",
        f"- F1 delta: {fmt(int8['deltas']['f1_delta'], 4)}",
        "",
        "## Deployment Interpretation",
        "",
        "- `ResNet-18` is the deployable baseline path for the current FastAPI runtime and now has measured PyTorch, ONNX FP32, and ONNX INT8 comparisons.",
        "- The hybrid winner remains a component-based serving policy. Its runtime signal is valuable for research comparison, but it is not exported as one unified ONNX artifact in this pass.",
        (
            "- PatchCore and WinCLIP remain native/runtime-bound components in this repository. "
            "The quantization benchmark is deliberately limited to the baseline model where the deployment path is already clean."
        ),
        "",
    ]

    write_markdown("\n".join(report), REPORT_PATH)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
