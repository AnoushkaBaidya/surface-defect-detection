# ResNet-18 Baseline Report

## Aggregate Metrics

| AUROC | F1 | Precision | Recall | FNR | Latency (ms) | Throughput (img/s) | Size (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8574 | 0.7499 | 0.9471 | 0.6905 | 0.3095 | 50.8 | 20.88 | 42.7 |

## Interpretation

ResNet-18 is the fastest and smallest benchmarked model, but it is also the weakest on recall and false-negative behavior.

## Hard Categories

The lowest-recall categories were:

- `toothbrush`: recall 0.0435
- `grid`: recall 0.2093
- `transistor`: recall 0.4000
- `hazelnut`: recall 0.5472
- `cable`: recall 0.5652

## Conclusion

ResNet-18 remains a useful supervised reference baseline, but it is not suitable as the final industrial detector in this benchmark.
