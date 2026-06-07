# Project Summary

## Business Problem

Surface-defect inspection systems need to catch defective parts while keeping manual review volume manageable. In this project, the main risk was not model training alone; it was whether the selected operating threshold would hold up when the visual acquisition domain changed.

False negatives represent escaped defects. False positives create operator review burden. The model-selection problem is therefore an operating-policy problem, not just a leaderboard-score problem.

## Dataset

The evaluation uses a temporal split:

- Source batch: training and validation data collected during the initial acquisition period
- Temporal-shift batch: images collected approximately one month later under changed acquisition conditions

The later batch exhibited systematic visual drift relative to the source data, including darker illumination, lower contrast, and reduced sharpness. These shifts were not caused by label changes, but by changes in the imaging domain over time.

The full inspection dataset and raw retained images are not included. The repo keeps saved prediction artifacts from a retained evaluation sample so the evaluation summary can be regenerated without shipping raw inspection images.

The earlier model performed well on the source distribution but degraded under the temporal-shift batch because its operating threshold and learned feature distribution were calibrated to the original imaging conditions. The evaluation therefore focuses not only on model accuracy, but on threshold robustness under realistic domain shift.

## Baseline Failure

The original ResNet18 classifier had strong ranking behavior but an unsafe operating threshold:

| Model / Policy | Precision | Recall | F1 | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| Original ResNet18 threshold | 1.0000 | 0.6979 | 0.8221 | 0 | 561 |

This is a common ML failure mode: high AUROC did not guarantee a safe threshold. The model was too conservative under domain shift.

## Final Model Decision

Threshold tuning changed the result substantially:

| Model / Policy | Precision | Recall | F1 | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| Tuned ResNet18 threshold | 0.9394 | 0.9521 | 0.9457 | 114 | 89 |

The selected policy is a ResNet18 classifier with an EfficientAD anomaly safety gate:

| Model / Policy | Precision | Recall | F1 | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| ResNet18 + EfficientAD safety gate | 0.9376 | 0.9553 | 0.9464 | 118 | 83 |

This hybrid catches six additional defects compared with tuned ResNet18 while adding four false positives in the held-out evaluation. That is a small improvement, so the tradeoff is explicit:

- Use tuned ResNet18 alone when operational simplicity is the priority.
- Use the hybrid policy when a small recall gain is worth the extra model artifact, threshold, latency, and monitoring surface.

## Implementation

The project turns model-selection work into a runnable service:

- ONNX Runtime inference wrappers for ResNet18 and EfficientAD
- FastAPI service with `/health`, `/predict`, and `/metrics`
- Dockerfile for containerized serving
- GitHub Actions for tests, linting, formatting, and Docker build validation
- Cloud Run instructions
- Load-test utilities for API serving checks
- Drift monitoring based on image-quality statistics
- Retraining trigger scaffolding based on drift and human label review

## Lessons

1. AUROC is not enough for inspection readiness.
2. Thresholds must be selected on representative shifted data.
3. Domain shift should be monitored as an input-data problem and a score-distribution problem.
4. Anomaly models can catch additional misses, but often add review burden.
5. A simpler model can be the better baseline if the hybrid gain is too small.
6. Retraining should be triggered through reviewable manifests, not automatic blind retraining.

## Where To Read More

- [Model selection report](reports/model_selection_report.md)
- [ResNet18 failure analysis](reports/resnet18_failure_analysis.md)
- [Anomaly model analysis](reports/anomaly_model_analysis.md)
- [Final hybrid policy report](reports/final_hybrid_policy_report.md)
- [Drift and retraining](DRIFT_AND_RETRAINING.md)
