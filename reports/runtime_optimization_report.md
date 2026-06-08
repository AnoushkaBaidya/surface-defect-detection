# Runtime Optimization Report

## Hardware Context

- CPU target: `Intel64 Family 6 Model 186 Stepping 3, GenuineIntel`
- Batch size: `1`
- Scope: MVTec AD bottle held-out benchmark artifacts

## Raw Latency Comparison

| Path | Size (MB) | Mean Latency (ms) | P95 Latency (ms) | Throughput (img/s) | F1 | Recall | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 baseline | 42.72 | 57.97 | 70.69 | 17.25 | 0.9677 | 0.9375 | 0.0625 |
| PatchCore + WinCLIP k=4 hybrid | 689.13 | 41720.29 | 41654.52 | 0.02 | 0.9773 | 0.9823 | 0.0177 |

The hybrid latency is reported as a sequential component policy (`PatchCore` + `WinCLIP k=4`) using the bottle-category artifact timings. The quality snapshot comes from the public final-selection comparison CSV.
This report is meant to answer a deployment question, not just a model-quality question: which result is strong enough to serve, and which result is better kept as a research reference.

## ONNX FP32 Export Comparison

| Path | Size (MB) | Mean Latency (ms) | P95 Latency (ms) | Throughput (img/s) | F1 | Recall | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 42.72 | 73.14 | 85.40 | 13.67 | 0.8276 | 1.0000 | 0.0000 |
| ONNX FP32 | 42.63 | 31.67 | 39.07 | 31.58 | 0.8276 | 1.0000 | 0.0000 |

- Mean defect-probability parity delta: 0.000000
- Max defect-probability parity delta: 0.000000

## INT8 Quantization Comparison

| Path | Size (MB) | Mean Latency (ms) | P95 Latency (ms) | Throughput (img/s) | F1 | Recall | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX FP32 | 42.63 | 34.23 | 45.56 | 29.21 | 0.8276 | 1.0000 | 0.0000 |
| ONNX INT8 | 10.71 | 377.22 | 405.12 | 2.65 | 0.8276 | 1.0000 | 0.0000 |

- Size reduction: 31.92 MB (74.88%)
- Mean latency delta: 342.99 ms
- P95 latency delta: 359.56 ms
- F1 delta: 0.0000

The INT8 result is still useful even though it is slower on this CPU-local setup. It shows that quantization reduced artifact size substantially, but that this hardware and runtime combination does not automatically convert smaller weights into lower latency.

## Deployment Interpretation

- `ResNet-18` is the deployable baseline path for the current FastAPI runtime and now has measured PyTorch, ONNX FP32, and ONNX INT8 comparisons.
- The hybrid winner remains a component-based serving policy. Its runtime signal is valuable for research comparison, but it is not exported as one unified ONNX artifact in this pass.
- PatchCore and WinCLIP remain native/runtime-bound components in this repository. The quantization benchmark is deliberately limited to the baseline model where the deployment path is already clean.
- The engineering decision from these measurements is straightforward: keep the hybrid policy as the quality upper bound, and keep the lighter baseline runtime path for CPU-local serving unless a different hardware target changes the tradeoff.
