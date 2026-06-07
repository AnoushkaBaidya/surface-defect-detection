# Model Card

## Model Overview

This repository serves a selective surface-defect detection policy:

- Primary model: ResNet18 binary classifier exported to ONNX
- Safety gate: EfficientAD anomaly model exported to ONNX
- Policy: flag as `defective` if ResNet18 exceeds its threshold or, when ResNet18 does not fire, EfficientAD exceeds its safety-gate threshold

The policy is intended for image-level defect screening.

## Intended Use

The model is designed for surface-defect inspection workflows where the goal is to classify images as:

- `normal`
- `defective`

It is not intended to replace quality assurance approval without environment-specific validation, review-capacity analysis, and monitoring.

## Inputs

Endpoint: `POST /predict`

Input format:

- Multipart image upload
- Supported image types depend on Pillow decoding, typically PNG/JPEG/BMP
- Images are converted to RGB and resized for model inference

Preprocessing:

- ResNet18 input size: `224 x 224`
- EfficientAD input size: `224 x 224`
- ResNet18 uses ImageNet-style normalization
- EfficientAD uses tensor conversion expected by the exported score model

## Outputs

The API returns:

- request ID
- model name and version
- prediction: `normal` or `defective`
- ResNet18 defect score and threshold
- EfficientAD normalized anomaly score and threshold, when evaluated
- decision reason
- latency breakdown

Example decision reasons:

- `resnet18_score_above_threshold`
- `efficientad_high_confidence_gate`
- `all_scores_below_threshold`

## Thresholds

| Component | Threshold | Role |
|---|---:|---|
| ResNet18 | 0.002 | Primary tuned classifier threshold |
| EfficientAD | 0.30 | Strict secondary safety-gate threshold |

The EfficientAD threshold is stricter than its standalone best-F1 threshold because it is used as a secondary gate, not as the primary detector. The goal is to catch a small number of additional ResNet18 misses without taking on the full false-positive load of standalone anomaly detection.

## Metrics

Held-out image-level evaluation:

| Policy | Precision | Recall | F1 | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| Original ResNet18 threshold | 1.0000 | 0.6979 | 0.8221 | 0 | 561 |
| Tuned ResNet18 threshold | 0.9394 | 0.9521 | 0.9457 | 114 | 89 |
| ResNet18 + EfficientAD safety gate | 0.9376 | 0.9553 | 0.9464 | 118 | 83 |

The selected hybrid provides a modest recall improvement over tuned ResNet18. Tuned ResNet18 remains the simpler single-model baseline.

## Limitations

- The full inspection dataset is not included in this repository.
- Evaluation is image-level classification only; held-out segmentation masks were not available.
- The model should be recalibrated before use on a new camera, lighting setup, part family, or defect taxonomy.
- The EfficientAD gate adds complexity and should be justified by review-capacity and escaped-defect targets.
- The model can be sensitive to brightness, contrast, sharpness, and acquisition drift.
- Metrics are from a held-out evaluation setting, not a guarantee of future behavior.

## Monitoring Recommendations

Monitor:

- input brightness, contrast, and sharpness
- prediction rate and defect-rate estimates
- ResNet18 score distribution
- EfficientAD score distribution
- false-positive review load
- sampled false-negative audits from human review
- latency and error rate

## Ethical and Safety Notes

This system is designed for industrial quality screening, not human-subject decision making. Safety risk is still real: a false negative can allow a defective part to pass inspection. The system should support human review, traceable logs, versioned thresholds, rollback, and periodic validation before use.
