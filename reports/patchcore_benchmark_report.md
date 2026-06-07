# PatchCore Benchmark Report

## Aggregate Metrics

| Classification AUROC | F1 | Recall | FNR | Segmentation AUROC | Segmentation F1 | Latency (ms) | Size (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9823 | 0.9710 | 0.9723 | 0.0277 | 0.9801 | 0.5817 | 1193.8 | 781.2 |

## Strengths

- best standalone classification AUROC
- best standalone recall and FNR
- best standalone segmentation quality
- strongest overall anomaly baseline

## Category Notes

PatchCore delivered the best classification AUROC in 12 of 15 MVTec categories and kept its lowest category recall at 0.9291 on `pill`.

## Limitation

Its deployment cost is high: large artifact size, low throughput, and the highest latency among the main anomaly baselines.

## Conclusion

PatchCore is the strongest pure anomaly detector in the repository and the quality anchor for the best hybrid system.
