# Benchmark Methodology

## Dataset Scope

The benchmark uses all 15 categories from MVTec AD:

- bottle
- cable
- capsule
- carpet
- grid
- hazelnut
- leather
- metal_nut
- pill
- screw
- tile
- toothbrush
- transistor
- wood
- zipper

## Evaluation Families

- supervised image classification with ResNet-18
- normal-only anomaly detection with PaDiM, PatchCore, and FastFlow
- zero/few-shot vision-language evaluation with WinCLIP
- weighted hybrid fusion between anomaly and VLM scores

## Metrics

### Classification

- AUROC
- F1
- precision
- recall
- false-negative rate
- false-positive rate
- false-negative count

### Segmentation

- segmentation AUROC
- segmentation F1

### Deployment-Oriented

- mean inference latency
- throughput
- model artifact size

## Selection Logic

The benchmark emphasizes:

1. recall
2. false-negative reduction
3. AUROC and F1 support
4. segmentation quality where available
5. latency and model size for feasibility analysis

This ordering reflects inspection reality: a model that misses fewer defects is more useful than one that looks strong on AUROC alone.

## Important Constraints

- this repo only uses validated MVTec benchmark outputs
- hybrid latency is not overstated where end-to-end timing is incomplete
- internal-domain validation is intentionally left out of this repo
- deferrals such as EfficientAD and AnomalyCLIP are documented explicitly
