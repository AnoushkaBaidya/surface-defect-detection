# Hybrid Architecture Analysis

## Fusion Goal

The hybrid experiments test whether an anomaly detector can remain the primary signal while a VLM contributes semantic context to improve missed-defect behavior.

## Best Result

| Model | Classification AUROC | F1 | Recall | FNR | Mean FN Count |
|---|---:|---:|---:|---:|---:|
| PatchCore + WinCLIP k=4 (0.7 / 0.3) | 0.9853 | 0.9773 | 0.9823 | 0.0177 | 1.53 |

## Why It Won

- PatchCore already provided the strongest anomaly backbone
- a conservative VLM contribution improved recall without overwhelming anomaly evidence
- heavier VLM weighting degraded results

## Design Lesson

The winning hybrid is not a VLM-led system. It is an anomaly-led system with a small semantic correction term.

## Limitation

Hybrid latency was not fully measured end-to-end in the same way as the standalone models, so the hybrid winner should be treated as a research-quality selection rather than a fully validated runtime claim.
