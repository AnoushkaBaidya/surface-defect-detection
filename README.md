# Surface Defect Detection

Industrial surface defect detection benchmark focused on **MVTec AD**, comparing supervised CNN baselines, anomaly detection methods, vision-language models, and hybrid score-fusion policies under shared evaluation and deployment tradeoffs.

## Tech Stack

PyTorch · ONNX Runtime · FastAPI · Docker · MLflow · Ray Data · GitHub Actions · Python

Companion branches in the broader project history also cover Docker deployment and drift-monitoring extensions; this branch centers the MVTec benchmark and its deployable runtime path.

## What This Covers

- Model training and A/B evaluation of `ResNet-18` versus the final hybrid serving policy
- INT8 quantization with measured latency, throughput, and model-size benchmarking
- Production-oriented serving with FastAPI and ONNX Runtime on the baseline path
- Experiment tracking with MLflow and published run artifacts in `mlruns/`
- Distributed image preprocessing with Ray Data for dataset preparation workflows
- CI-ready Python project structure with unit tests and lintable scripts

## Project Overview

Industrial inspection systems rarely have abundant labeled defect images. In practice, teams often have many normal samples, sparse defect coverage, and strong pressure to minimize missed defects. This repository studies that setting directly by benchmarking multiple model families across all 15 MVTec AD categories.

The central question is not only which model achieves the highest AUROC, but which approach best balances:

- missed-defect risk
- localization quality
- latency and model size
- interpretability of tradeoffs

## Research Focus

The benchmark is organized around four model families:

- **ResNet-18 supervised baseline** for a conventional defect classifier reference point
- **Anomaly detection baselines** using PaDiM, PatchCore, and FastFlow
- **Vision-language evaluation** using WinCLIP in zero-shot and few-shot settings
- **Hybrid anomaly + VLM fusion** to test whether semantic prompts can improve recall without replacing anomaly scoring

## Key Features

- Benchmarking across all 15 MVTec AD categories
- Image-level classification and pixel-level segmentation analysis
- ResNet-18, PaDiM, PatchCore, FastFlow, and WinCLIP evaluation
- Hybrid score-fusion experiments across anomaly and VLM signals
- Recall and false-negative-rate prioritization
- Latency, throughput, and model-size tradeoff analysis
- Reports and summaries generated from the benchmark outputs in this repo
- Experiment runs tracked with MLflow — see `reports/mlflow_experiment_comparison.png` and `mlruns/`
- Ray Data preprocessing pipeline for distributed image preparation workflows

## Results Overview

### A/B Policy Evaluation Results

### Primary model positioning

| Model | Classification AUROC | F1 | Recall | FNR | Segmentation AUROC | Mean Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 | 0.8574 | 0.7499 | 0.6905 | 0.3095 | N/A | 50.8 |
| PaDiM | 0.8881 | 0.9104 | 0.9484 | 0.0516 | 0.9596 | 447.1 |
| PatchCore | 0.9823 | 0.9710 | 0.9723 | 0.0277 | 0.9801 | 1193.8 |
| FastFlow | 0.9336 | 0.9449 | 0.9606 | 0.0394 | 0.9715 | 502.7 |
| WinCLIP k=1 | 0.8976 | 0.9239 | 0.9597 | 0.0403 | 0.9057 | 19553.3 |
| PatchCore + WinCLIP k=4 (0.7 / 0.3) | 0.9853 | 0.9773 | 0.9823 | 0.0177 | N/A | not fully measured |

### Headline findings

- **Best pure anomaly detector:** PatchCore
- **Best production-feasible anomaly model:** FastFlow
- **Best VLM setting in this repo:** WinCLIP few-shot variants, with recall benefits but large latency costs
- **Best aggregate model quality:** PatchCore + WinCLIP k=4 with 0.7 anomaly / 0.3 VLM weighting
- **ResNet-18 ONNX FP32 runtime result:** mean bottle-category latency dropped from 73.14 ms to 31.67 ms with no F1 change
- **ResNet-18 ONNX INT8 runtime result:** model size dropped from 42.63 MB to 10.71 MB (74.88%) with no F1 change, but latency increased on this CPU-local setup
- **Hybrid policy runtime result:** PatchCore + WinCLIP k=4 measured 41720.29 ms mean end-to-end as a sequential bottle-category serving policy

The repository separates the research winner from the serving winner on purpose. The hybrid policy is the best quality result in the benchmark, while the deployable runtime path remains the lighter `ResNet-18` baseline with measured ONNX export behavior.

### What matters most here

- Supervised inspection underperformed on unseen-defect risk
- Normal-only anomaly detection closed most of the recall gap
- Hybrid fusion improved recall further, but only when anomaly scoring remained dominant
- VLMs were useful as auxiliary signals, not as standalone industrial winners in this local benchmark

## Architecture

```text
MVTec AD
   |
   +--> dataset profiling
   |
   +--> supervised baseline branch
   |       |
   |       +--> ResNet-18 training
   |       +--> image-level evaluation
   |
   +--> anomaly detection branch
   |       |
   |       +--> PaDiM
   |       +--> PatchCore
   |       +--> FastFlow
   |       +--> segmentation + classification evaluation
   |
   +--> VLM branch
   |       |
   |       +--> WinCLIP zero-shot / few-shot
   |
   +--> hybrid branch
           |
           +--> score fusion
           +--> recall / FNR analysis
           +--> final model selection
```

## System Flow

```text
download data -> profile categories -> preprocess with Ray Data
             -> run baseline models -> normalize metrics -> compare families
             -> analyze latency / size / recall tradeoffs
             -> select final model positioning
```

## Technical Stack

- Python 3.11
- PyTorch and torchvision
- Anomalib
- timm
- OpenCV
- Hugging Face / CLIP-adjacent tooling
- MLflow
- Ray Data
- pandas and matplotlib
- scikit-learn evaluation tooling

## Benchmark Methodology

- All 15 MVTec AD categories are evaluated
- Classification and segmentation metrics are separated explicitly
- Recall and false-negative rate are prioritized over AUROC alone
- Supervised and anomaly methods are compared under one reporting contract
- VLM results are treated as benchmark experiments, not production claims
- Hybrid fusion is evaluated conservatively with anomaly-dominant weighting
- Latency, throughput, and model artifact size are included when measured

More detail lives in:

- `docs/benchmark_methodology.md`
- `docs/benchmark_design_decisions.md`
- `docs/engineering_tradeoffs.md`

## Data Pipeline

Ray Data is used in `scripts/ray_data_pipeline.py` for distributed image preprocessing and optional augmentation over local MVTec image directories. The pipeline is intentionally lightweight and keeps the research flow reproducible on a single machine before scaling out.

## Repository Structure

```text
assets/        charts and sample visuals
configs/       experiment and dataset configuration
deployment/    optional scaffolding kept separate from the MVTec benchmark core
docker/        optional container scaffolding
docs/          technical documentation
reports/       polished benchmark reports
scripts/       dataset, training, benchmarking, and data-pipeline utilities
src/           reusable library code
tests/         unit and integration tests
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-ml.txt
pip install anomalib
python scripts/download_mvtec.py
python scripts/profile_dataset.py
python scripts/ray_data_pipeline.py --input-dir data/mvtec_anomaly_detection/bottle/train/good --output-dir artifacts/ray_pipeline_preview
```

Recommended next reads:

- `docs/local_setup.md`
- `reports/final_model_selection_report.md`
- `reports/engineering_findings_report.md`
- `reports/runtime_optimization_report.md`

## Visual Assets

- `assets/benchmark_charts/classification_auroc_comparison.png`
- `assets/benchmark_charts/recall_comparison.png`
- `assets/benchmark_charts/fnr_comparison.png`
- `assets/benchmark_charts/latency_vs_auroc.png`
- `assets/dataset_samples/bottle_normal_samples.png`

## Open Questions

- remeasure hybrid latency end-to-end on a unified optimized runtime surface
- test whether the hybrid gain holds under stricter threshold selection
- inspect why ResNet-18 wins isolated categories like `screw` and `zipper`
- compare FastFlow and PatchCore under tighter runtime constraints

## Engineering Lessons

- Recall and FNR matter more than AUROC alone in inspection settings
- Supervised defect classifiers are brittle when defect coverage is incomplete
- PatchCore is extremely strong, but expensive to deploy
- FastFlow offers a better quality-to-latency tradeoff than PatchCore
- VLMs bring semantic signal, but they are not free from serious throughput costs
- Hybrid systems help only when fusion is disciplined and well-scoped
