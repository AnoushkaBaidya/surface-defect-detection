# Dataset Contract

## Data Included

This repository does not include raw inspection images. It includes saved prediction artifacts from a retained evaluation sample so the thresholding and report-generation path can be checked without exposing the full dataset.

Saved retained-sample artifacts:

- `reports/predictions/retained_sample_resnet18_predictions.csv`
- `reports/predictions/retained_sample_resnet18_threshold.json`
- `reports/predictions/retained_sample_resnet18_threshold_sweep.csv`
- `reports/reproducibility/retained_sample_resnet18_metrics.json`
- `reports/reproducibility/retained_sample_resnet18_score_monitoring.json`
- `reports/reproducibility/retained_sample_resnet18_summary.md`

If local retained images are available under `data_samples/`, the same scripts can rebuild the manifest, splits, and predictions. Those images are intentionally not part of the repository.

## What is not included

The full inspection dataset, raw retained images, training-time annotations outside the saved artifacts, review logs, and organization-specific metadata are intentionally omitted.

Not committed:

- full raw image history
- raw retained sample images
- operator review labels beyond the saved retained-sample artifacts
- internal checkpoint lineage before export
- organization-specific cloud settings and secrets

## Manifest contract

When local retained images are available, the sample pipeline generates `reports/datasets/retained_sample_manifest.csv` with the following columns:

| Column | Type | Description |
|---|---|---|
| `dataset_name` | string | Logical dataset group such as `baseline_oct14` or `production_nov8` |
| `capture_batch` | string | Human-readable capture batch identifier |
| `label` | string | `normal` or `defective` |
| `label_id` | int | `0` for normal, `1` for defective |
| `relative_path` | string | Repo-relative image path |
| `is_private` | bool | Whether the sample points to a private-only source |
| `is_simulated` | bool | Whether the sample is synthetic or derived |

Machine-readable schema:

- [configs/pipeline/retained_sample_manifest.schema.json](../configs/pipeline/retained_sample_manifest.schema.json)

Committed sample summary:

- [reports/datasets/retained_sample_manifest_summary.json](../reports/datasets/retained_sample_manifest_summary.json)

## Split logic

The sample split logic lives in [scripts/create_retained_sample_split.py](../scripts/create_retained_sample_split.py).

Rules:

1. `baseline_oct14` is the source domain.
2. `production_nov8` is the held-out evaluation domain.
3. Train/validation are stratified from `baseline_oct14`.
4. Test is the full `production_nov8` sample.

This mirrors the temporal split used in the project without exposing the full dataset.

## Reproducibility commands

Regenerate report artifacts from saved retained-sample predictions:

```bash
python -m scripts.reproduce_retained_sample_pipeline --from-saved-predictions
```

If local retained images are available, regenerate the full sample pipeline:

```bash
python -m scripts.reproduce_retained_sample_pipeline
```
