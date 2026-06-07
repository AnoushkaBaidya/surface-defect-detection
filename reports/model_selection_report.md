# Model Selection Report

## Executive Summary

This project was not a simple model-accuracy exercise. The central problem was selecting an operating policy under domain shift, false-negative risk, review-capacity limits, and service constraints.

The original ResNet18 classifier had strong ranking performance but failed at the selected operating threshold. Threshold calibration recovered most of the missed recall. Anomaly models were then evaluated as safety-net signals, but their standalone false-positive load was too high. The selected policy is therefore a selective hybrid: tuned ResNet18 as the primary detector, with EfficientAD as a strict anomaly safety gate.

## Decision Flow

1. Evaluate source-to-held-out domain shift.
2. Diagnose the ResNet18 false-negative failure mode.
3. Benchmark anomaly models under the same operating constraints.
4. Select an operating policy that balances escaped defects, review load, and engineering complexity.

## Why The Baseline Failed

The original ResNet18 threshold produced no false positives but missed too many defects:

| Policy | Precision | Recall | F1 | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| Original ResNet18 threshold | 1.0000 | 0.6979 | 0.8221 | 0 | 561 |

The issue was not that ResNet18 had no useful signal. The issue was that the selected threshold did not transfer safely under shifted inspection conditions. The held-out batch was darker, lower contrast, and less sharp than the source batch, and many missed defects received normal-like scores.

## Tuned Supervised Baseline

Threshold calibration changed the operating point:

| Policy | Precision | Recall | F1 | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| Tuned ResNet18 threshold | 0.9394 | 0.9521 | 0.9457 | 114 | 89 |

This became the strongest single-model baseline. It reduced escaped defects substantially while keeping review load much lower than standalone anomaly models.

## Anomaly Model Findings

PatchCore, FastFlow, and EfficientAD were evaluated as anomaly-detection alternatives. They were useful diagnostically because they could catch defects that the original ResNet18 threshold missed, but their standalone operating points created hundreds of false positives.

| Model / Operating Point | Precision | Recall | F1 | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| FastFlow, best F1 | 0.8109 | 0.9607 | 0.8795 | 416 | 73 |
| PatchCore, best F1 | 0.8088 | 0.9225 | 0.8619 | 405 | 144 |
| EfficientAD, best F1 | 0.8031 | 0.9666 | 0.8773 | 440 | 62 |

The interpretation is that anomaly models are better as safety-net signals than as standalone classifiers in this setting.

## Final Policy

| Policy | Precision | Recall | F1 | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| ResNet18 + EfficientAD safety gate | 0.9376 | 0.9553 | 0.9464 | 118 | 83 |

The hybrid policy catches six additional defects compared with tuned ResNet18 while adding four false positives. That improvement is modest, so the tradeoff is explicit:

- If operational simplicity matters most, use tuned ResNet18.
- If reducing escaped defects is worth the extra model artifact and monitoring surface, use the ResNet18 + EfficientAD safety gate.

## Lesson

The strongest finding is that AUROC was not enough. The core question was not "which model ranks best?" It was "which operating policy catches defects at an acceptable review cost under shifted conditions?"

That is the reason the repo includes model serving, drift monitoring, retraining-trigger scaffolding, CI checks, and service documentation rather than only training notebooks.
