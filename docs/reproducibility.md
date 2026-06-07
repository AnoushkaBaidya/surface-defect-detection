# Reproducibility

## Environment

- Python 3.11
- dependency installation through `requirements-ml.txt`
- development tooling through `requirements-dev.txt`

## Dataset Preparation

1. Download or manually place the MVTec AD archive.
2. Run `python scripts/download_mvtec.py`.
3. Run `python scripts/profile_dataset.py`.

## Benchmark Execution Order

1. `python scripts/create_resnet18_splits.py`
2. `python scripts/run_resnet18_benchmark.py`
3. `python scripts/run_padim_benchmark.py`
4. `python scripts/run_patchcore_benchmark.py`
5. `python scripts/run_fastflow_benchmark.py`
6. `python scripts/run_winclip_benchmark.py`
7. `python scripts/run_score_exports.py`
8. `python scripts/build_hybrid_benchmark.py`
9. `python scripts/build_benchmark_summary.py`
10. `python scripts/create_final_model_selection_report.py`

## Validation

- `python scripts/validation/validate_repository_setup.py`
- `python scripts/validation/validate_configs.py`
- `python scripts/validation/validate_imports.py`
- `pytest`

## Reproducibility Notes

- Random seeds are set inside training scripts where deterministic behavior is expected.
- Some downstream library operations may still vary across hardware, CUDA versions, or dependency versions.
- MLflow is used for run tracking, parameter logging, and artifact persistence.
