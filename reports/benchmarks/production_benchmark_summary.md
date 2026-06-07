# Production Benchmark Summary

## Scope

This report summarizes committed benchmark artifacts for ONNX Runtime model latency, FastAPI serving latency, and Cloud Run load-test throughput.

These results support service and throughput checks. They should not be interpreted as model-accuracy validation on new real-world images.

## ONNX Runtime Latency

| Model | Mean ms | p50 ms | p95 ms | p99 ms | Runs |
|---|---:|---:|---:|---:|---:|
| resnet18_onnx | 26.09 | 25.90 | 27.59 | 29.62 | 200 |
| efficientad_onnx | 186.00 | 180.13 | 228.73 | 250.50 | 100 |

## Local FastAPI Latency

- Requests tested: `100`
- EfficientAD invocations: `89`

| Metric | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| API total | 200.87 | 209.54 | 265.25 | 285.93 |
| ResNet18 | 28.82 | 28.22 | 32.55 | 37.71 |
| EfficientAD | 190.46 | 181.44 | 235.43 | 249.24 |

## Resume Claim Guidance

Supported claims from committed artifacts should use the exact reported numbers above.

Do not claim 400K+ or 500K+ requests/day unless the committed load-test output shows that estimate.
