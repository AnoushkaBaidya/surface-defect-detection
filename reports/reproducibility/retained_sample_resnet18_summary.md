# Retained Sample Evaluation

- Model artifact: `models\resnet18_internal.onnx`
- Threshold: `0.0030`
- Accuracy: `0.9383`
- Precision: `0.9256`
- Recall: `0.9533`
- F1: `0.9392`
- False positives: `23`
- False negatives: `14`

## Score Monitoring Summary

| Group | Mean | Std | P05 | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| all | 0.3198 | 0.4236 | 0.0002 | 0.0056 | 0.9996 |
| normal | 0.0019 | 0.0100 | 0.0001 | 0.0004 | 0.0084 |
| defective | 0.6376 | 0.3959 | 0.0032 | 0.8061 | 0.9998 |