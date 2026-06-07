# PaDiM Benchmark Report

## Aggregate Metrics

| Classification AUROC | F1 | Recall | FNR | Segmentation AUROC | Segmentation F1 | Latency (ms) | Size (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8881 | 0.9104 | 0.9484 | 0.0516 | 0.9596 | 0.4664 | 447.1 | 561.8 |

## Interpretation

PaDiM strongly outperformed the supervised baseline on recall and defect miss rate, validating the move toward normal-only anomaly detection.

## Positioning

- good classical anomaly benchmark
- solid segmentation support
- weaker than PatchCore and FastFlow on overall aggregate quality

## Notable Category Result

PaDiM was the strongest classification-AUROC model on `carpet`, showing that texture-heavy categories can favor different anomaly behavior than the aggregate winner.
