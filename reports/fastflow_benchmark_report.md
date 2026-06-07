# FastFlow Benchmark Report

## Aggregate Metrics

| Classification AUROC | F1 | Recall | FNR | Segmentation AUROC | Segmentation F1 | Latency (ms) | Throughput (img/s) | Size (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9336 | 0.9449 | 0.9606 | 0.0394 | 0.9715 | 0.5387 | 502.7 | 2.05 | 268.7 |

## Why FastFlow Matters

FastFlow does not beat PatchCore on raw quality, but it materially improves the runtime/size profile:

- lower latency than PatchCore
- higher throughput than PatchCore
- much smaller model artifact size than PatchCore

## Best-Use Position

FastFlow is the anomaly model with the cleanest runtime/size tradeoff in this benchmark and the best compromise between quality and overhead.
