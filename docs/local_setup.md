# Local Setup

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-ml.txt
pip install anomalib
```

## Data

```bash
python scripts/download_mvtec.py
python scripts/profile_dataset.py
```

## Running Benchmarks

The repository contains utilities for:

- supervised baseline training
- anomaly benchmarking
- WinCLIP evaluation
- report generation

Start with the dataset profile and then run the benchmark utilities that match the model family you want to inspect.

The active workflow in this repo is the MVTec benchmark. Deployment, API, and container folders are optional scaffolding and are not required for the research flow.

## Validation

```bash
pytest
ruff check .
```

## Notes

- local raw data and model artifacts are intentionally ignored by Git
- hybrid latency claims should be remeasured before stronger runtime conclusions
