# Surface Defect Detection — Industrial Inspection ML Platform

Two end-to-end ML engineering projects for automated visual inspection in manufacturing environments. Each branch is a self-contained system covering model evaluation, production serving, experiment tracking, drift monitoring, and deployment infrastructure.

---

## Tech Stack

PyTorch · ONNX Runtime · FastAPI · Docker · Kubernetes · MLflow · Ray Data · Prometheus · GitHub Actions · Python · Ruff · Black · pytest

---

## Projects at a Glance

| | [MVTec AD Benchmark](#project-1-industrial-anomaly-detection-benchmark--inference-platform) | [Internal Dataset Pipeline](#project-2-production-defect-inspection-service--mlops-pipeline) |
|---|---|---|
| **Branch** | `mvtec-dataset-pipeline` | `internal-dataset-pipeline` |
| **Dataset** | MVTec AD (public, 15 categories) | Proprietary inspection images |
| **Models** | ResNet-18, PaDiM, PatchCore, FastFlow, WinCLIP | ResNet-18, EfficientAD |
| **Winner** | PatchCore + WinCLIP hybrid | ResNet-18 + EfficientAD safety gate |
| **Recall** | 98.2% | 95.5% |
| **False-Negative Rate** | 1.8% | 4.5% |
| **Serving** | FastAPI + ONNX Runtime | FastAPI + ONNX Runtime |
| **Infra** | Kubernetes · Docker · GCP Cloud Run | Kubernetes · Docker · GCP Cloud Run |

---

## Project 1: Industrial Anomaly Detection Benchmark & Inference Platform

**→ Branch: [`mvtec-dataset-pipeline`](../../tree/mvtec-dataset-pipeline)**

### Problem

Industrial surface inspection systems fail quietly: a model that looks good on AUROC can still miss critical defects when visual acquisition conditions shift across categories, lighting setups, or production lines. The challenge is not just training a model — it is designing an evaluation system that reveals which model actually minimizes missed defects at a deployable operating point.

### What I Built

A config-driven benchmarking platform that evaluates four model families — supervised CNN, anomaly detection, VLM, and hybrid fusion — across all 15 MVTec AD categories under a single shared evaluation contract. The benchmark prioritizes recall and false-negative rate over AUROC, reflecting the real cost asymmetry of industrial inspection where a missed defect is far more expensive than a false alarm.

The selected serving policy — a PatchCore + WinCLIP hybrid at fusion weights 0.7/0.3 — is packaged into a FastAPI inference runtime with pixel-level anomaly map output, INT8 quantization, Kubernetes deployment, and MLflow experiment tracking.

### Key Results

| Model / Policy | Recall | False-Negative Rate | Mean Latency |
|---|---|---|---|
| ResNet-18 baseline | 69.1% | 30.9% | — |
| FastFlow | 96.1% | 3.9% | 502.7 ms |
| PatchCore | — | — | 1193.8 ms |
| **PatchCore + WinCLIP hybrid (winner)** | **98.2%** | **1.8%** | — |

- ~94% reduction in false-negative rate vs supervised baseline
- ~29 percentage-point recall gain over ResNet-18
- FastFlow validated as runtime-optimal anomaly candidate: ~58% lower latency than PatchCore
- INT8 quantization via ONNX Runtime: reduced model size with <Y% F1 degradation *(fill in after benchmark run)*

### Engineering Decisions

**Why recall over AUROC?** AUROC measures ranking quality across all thresholds and hides the operating behavior that matters in production. A model with AUROC 0.97 can still miss 30% of defects at the threshold a practitioner would actually use. All model selection decisions in this benchmark are made on recall, false-negative rate, and F1 at a calibrated operating threshold.

**Why a hybrid policy?** No single model family dominates across all 15 MVTec categories. Anomaly models (PatchCore) generalize well but are too slow for real-time lines. VLMs (WinCLIP) add semantic reasoning that pure distance-based methods miss. The hybrid fuses both signals at inference time without retraining either component.

**Why FastFlow as the runtime baseline?** PatchCore wins on recall but at 1193.8 ms mean latency it is not viable for high-throughput lines. FastFlow achieves 96.1% recall at 502.7 ms — a ~58% latency reduction with acceptable quality tradeoff. This is documented explicitly in the benchmark so the deployment tradeoff is auditable, not ad hoc.

**Why component-level quantization, not a single hybrid export?** The hybrid winner is a serving policy built from two model components plus score fusion — not a single neural artifact. Pretending otherwise would produce a misleading benchmark. ResNet-18 and FastFlow are exported and quantized individually; the hybrid is benchmarked end-to-end at the policy level.

### Architecture

```
Ray Data Pipeline
└── Distributed image loading + preprocessing (15 MVTec categories)
    └── Augmentation + normalization
        └── Per-category train/val/test splits

Benchmark Harness (MLflow-tracked)
├── ResNet-18 (supervised baseline)
├── PaDiM (patch distribution modeling)
├── PatchCore (nearest-neighbor anomaly detection)
├── FastFlow (normalizing flow, runtime-optimized)
└── WinCLIP (zero-shot VLM)
    └── Hybrid fusion: PatchCore (0.7) + WinCLIP (0.3)

Inference Runtime
└── FastAPI /predict
    ├── ONNX Runtime (FP32 + INT8)
    ├── Pixel-level anomaly map output
    ├── Prometheus metrics
    └── /health endpoint

Deployment
├── Docker
├── Kubernetes (deployment.yaml · service.yaml · hpa.yaml)
└── GCP Cloud Run

CI/CD
└── GitHub Actions: Ruff · Black · config validation · pytest (20 tests)
```

### What's in the Branch

```
├── scripts/          Training, export, benchmark, and quantization scripts
├── app/              FastAPI inference service + ONNX model wrappers
├── configs/          Typed, validated experiment configurations
├── reports/          Model selection, benchmark, segmentation, and runtime reports
├── benchmark_outputs/ Per-category JSON results + quantization comparison
├── drift/            Drift detection + retraining trigger scaffolding
├── infra/kubernetes/ Deployment, service, and HPA manifests
├── .github/workflows/ CI pipeline
└── requirements-ml.txt Ray Data + ML training dependencies
```

**→ [View full project in `mvtec-benchmark` branch](../../tree/mvtec-benchmark)**

---

## Project 2: Production Defect Inspection Service & MLOps Pipeline

**→ Branch: [`internal-dataset-pipeline`](../../tree/internal-dataset-pipeline)**

### Problem

A supervised classifier trained on proprietary inspection images achieves near-perfect precision but misses 30% of defects at its default threshold — a failure mode AUROC entirely conceals. The task is to close that recall gap without sacrificing the precision that keeps operator review load manageable, then package the result into a production-ready serving stack with drift monitoring.

### What I Built

A selective serving policy that runs a threshold-calibrated ResNet-18 as the primary detector and routes non-flagged images through an EfficientAD anomaly safety gate. The hybrid reduces missed defects from 561 to 83 on the held-out evaluation set — an 85% reduction — while keeping precision above 93%. The policy is exported to ONNX Runtime, served via FastAPI, monitored with Prometheus and KS-statistic drift detection, and deployed with Docker and Kubernetes.

### Key Results — A/B Policy Evaluation

| Policy | Precision | Recall | F1 | False Positives | False Negatives |
|---|---|---|---|---|---|
| ResNet-18 (original threshold) | 1.000 | 69.8% | 0.822 | 0 | 561 |
| **ResNet-18 + EfficientAD gate (winner)** | **93.8%** | **95.5%** | **0.946** | **118** | **83** |

- 85% reduction in missed defects vs original threshold policy
- Threshold calibration recovered ~26 recall points with no new model training
- Hybrid policy adds 118 false positives — documented tradeoff, not ignored

### Engineering Decisions

**Why not just tune the threshold?** Tuning ResNet-18's threshold alone improves recall to ~95% but creates a recall floor: images where ResNet-18 scores are structurally low regardless of threshold won't be recovered. EfficientAD operates on a different feature space (reconstruction-based anomaly detection), catching defect types the classifier misses structurally.

**Why keep ResNet-18 as the primary gate?** Running EfficientAD on every image would increase latency and false positive rate on the majority class. The selective policy runs EfficientAD only when ResNet-18 does not flag — roughly 30% of traffic at the operating threshold — keeping overall latency acceptable and false positives bounded.

**Why KS statistics for drift detection?** Kolmogorov-Smirnov tests on image-quality features (brightness, contrast, sharpness, defect rate) are distribution-free and don't require labeled drift examples. They catch the acquisition condition shifts this system was designed to handle, without needing a supervised drift classifier.

### Architecture

```
Image Upload → FastAPI /predict
    ├── Preprocessing (normalize, resize)
    ├── ResNet-18 ONNX (primary classifier)
    │   ├── score ≥ 0.002 → defective (fast path)
    │   └── score < 0.002 → EfficientAD ONNX safety gate
    │       ├── score ≥ 0.30 → defective
    │       └── score < 0.30 → normal
    └── JSON response: label · scores · decision reason · latency breakdown

Monitoring
├── Prometheus metrics (/metrics)
├── Structured JSON logging
└── KS-statistic drift detection + retraining trigger scaffolding

Deployment
├── Docker
├── Kubernetes (deployment.yaml · service.yaml · hpa.yaml)
└── GCP Cloud Run (load tested at 1000 req, 4 concurrency)
```

### What's in the Branch

```
├── app/              FastAPI service + ONNX model wrappers
├── scripts/          Evaluation, threshold tuning, export, benchmark scripts
├── drift/            KS-statistic drift detection + retraining trigger
├── load_tests/       Async load testing + synthetic image generation
├── reports/          Model selection, failure analysis, hybrid policy, load test reports
├── infra/kubernetes/ Deployment, service, and HPA manifests
├── configs/          Model policy and calibration settings
├── models/           ONNX model artifacts for local demo
├── .github/workflows/ CI pipeline
└── MODEL_CARD.md     Model documentation with known limitations
```

**→ [View full project in `internal-dataset-pipeline` branch](../../tree/internal-dataset-pipeline)**

---

## Shared Engineering Standards

Both projects follow the same quality baseline:

- **Linting & formatting:** Ruff + Black, enforced in CI
- **Testing:** pytest with coverage across API endpoints, config validation, dataset splits, drift utilities, and serialization
- **Reproducibility:** Config-driven experiments, artifact manifests, and saved prediction artifacts so evaluation summaries regenerate without raw data
- **Documentation:** Architecture docs, dataset contracts, model cards, and deployment guides — claims map to measured outputs, not narrative
- **CI/CD:** GitHub Actions pipeline on every push

---

## Local Setup

Each branch has its own `requirements.txt`, `requirements-dev.txt`, and `requirements-ml.txt`. Clone and check out the branch you want:

```bash
git clone https://github.com/AnoushkaBaidya/surface-defect-detection.git
cd surface-defect-detection

# MVTec benchmark
git checkout mvtec-dataset-pipeline

# Internal dataset pipeline
git checkout internal-dataset-pipeline
```

Then follow the setup instructions in that branch's `README.md`.

---

