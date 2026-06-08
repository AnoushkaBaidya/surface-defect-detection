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
- runtime optimization benchmarking
- distributed image preprocessing with Ray Data
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
- runtime optimization utilities now include bottle-category raw latency, ONNX FP32 export, and ONNX INT8 quantization benchmarking for the ResNet18 deployment path
- hybrid runtime remains a component-based policy comparison rather than a unified exported model
- `scripts/ray_data_pipeline.py` provides a lightweight Ray Data preprocessing and augmentation path for local MVTec image directories
