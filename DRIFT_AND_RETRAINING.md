# Drift and Retraining

## Overview

The project includes lightweight drift monitoring and retraining-trigger scaffolding. The goal is not to retrain automatically. The goal is to detect when recent inspection data no longer resembles the baseline profile and create a reviewable retraining package.

## Baseline Profile

The baseline profile is built from a known reference image set. For each image, the drift pipeline extracts:

- brightness mean
- contrast standard deviation
- sharpness via Laplacian variance
- label, when available from a reviewed source

The extraction logic lives in `drift/compute_drift_features.py`; `drift/build_baseline_profile.py` is the command-line wrapper that builds baseline or recent-sample profiles from configured folders.

Command:

```bash
python drift/build_baseline_profile.py \
  --config drift/drift_config.yaml \
  --mode baseline
```

The output is written to:

```text
artifacts/drift/source_baseline_profile.json
```

`artifacts/` and `data_samples/` are intentionally ignored by Git so private data and local outputs are not committed.

## Recent-Sample Profile

A recent-sample profile is built with the same feature extraction logic:

```bash
python drift/build_baseline_profile.py \
  --config drift/drift_config.yaml \
  --mode production
```

The output is written to:

```text
artifacts/drift/heldout_production_profile.json
```

In a running inspection workflow, this profile would be generated from sampled recent inspection images and human-reviewed labels where available.

## Drift Detection

Drift detection compares baseline and recent-sample profiles using:

- relative change in feature means
- Kolmogorov-Smirnov test on feature distributions
- defect-rate change, when labels are available
- a minimum recent-sample count guard before recommending retraining

Command:

```bash
python drift/detect_drift.py
```

Outputs:

```text
artifacts/drift/drift_report.json
artifacts/drift/drift_report.md
```

Configured warning thresholds live in:

```text
drift/drift_config.yaml
```

Current monitored features:

| Feature | Why It Matters |
|---|---|
| Brightness mean | Lighting/camera exposure changes can move scores |
| Contrast standard deviation | Low contrast can hide subtle defects |
| Sharpness Laplacian variance | Blur can reduce defect visibility |
| Defect-rate change | Process changes or model behavior shifts can change review load |

## Retraining Trigger

The retraining trigger reads the latest drift report:

```bash
python drift/trigger_retraining.py
```

If drift exceeds configured thresholds, it writes:

```text
artifacts/retraining/retraining_trigger.json
artifacts/retraining/retraining_manifest.csv
```

The manifest is a candidate package, not an automatic model-training command. The trigger payload includes timestamp, drift reason, profile names, sample counts, drifted-feature count, and a human-label-review requirement.

## Human Labels

Retraining labels should come from:

- operator review decisions
- quality-engineering audits
- sampled over-review analysis
- sampled miss audit, where downstream inspection finds escaped defects
- controlled relabeling of uncertain cases

The local demo can infer labels from folder names for simulation, but this is not a substitute for reviewed label governance.

## Retraining Workflow

A realistic retraining workflow should be:

1. Detect drift.
2. Create a candidate manifest.
3. Review and validate labels.
4. Train a candidate model in a reproducible job.
5. Evaluate against the baseline, held-out data, and review-capacity targets.
6. Export model artifacts and update thresholds.
7. Run the updated service in a staging or test environment.
8. Run smoke tests and load tests.
9. Promote only if metrics and operational constraints are satisfied.

## Guardrails

Retraining should not happen blindly. Require:

- minimum sample count
- human-reviewed labels
- versioned data manifest
- reproducible training config
- comparison against the current model policy
- rollback plan
- documented threshold-selection rationale
