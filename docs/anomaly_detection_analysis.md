# Anomaly Detection Analysis

## Aggregate Results

| Model | Classification AUROC | F1 | Recall | FNR | Segmentation AUROC | Segmentation F1 |
|---|---:|---:|---:|---:|---:|---:|
| PaDiM | 0.8881 | 0.9104 | 0.9484 | 0.0516 | 0.9596 | 0.4664 |
| PatchCore | 0.9823 | 0.9710 | 0.9723 | 0.0277 | 0.9801 | 0.5817 |
| FastFlow | 0.9336 | 0.9449 | 0.9606 | 0.0394 | 0.9715 | 0.5387 |

## Findings

- PatchCore is the strongest pure anomaly model on classification and segmentation quality
- FastFlow retains strong quality while materially reducing latency and model size versus PatchCore
- PaDiM validates the anomaly-detection shift, but it trails PatchCore and FastFlow on overall robustness

## Category Behavior

PatchCore delivered the best classification AUROC in 12 of 15 categories. The remaining category leaders were:

- `carpet`: PaDiM
- `screw`: ResNet-18
- `zipper`: ResNet-18

Those isolated wins do not change the aggregate conclusion: anomaly detection is much more stable than the supervised baseline across the full benchmark.

## Practical Interpretation

- choose PatchCore when maximizing benchmark quality is the goal
- choose FastFlow when deployment feasibility matters more
- use PaDiM as a useful classical baseline, not the final recommendation
