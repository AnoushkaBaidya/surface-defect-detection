# Architecture

## Objective

The repository is organized as a benchmarking system rather than a single-model demo. It evaluates multiple model families under a common metric contract and then uses those results to compare tradeoffs directly.

## Pipeline

```text
MVTec AD
   |
   +--> data download and profiling
   |
   +--> supervised baseline evaluation
   |
   +--> anomaly detection benchmarking
   |
   +--> vision-language evaluation
   |
   +--> hybrid fusion experiments
   |
   +--> aggregate comparison and final selection
```

## Main Components

- `scripts/` contains runnable benchmark and reporting utilities
- `src/` contains reusable evaluation and modeling code
- `configs/` stores dataset and experiment configuration
- `reports/` summarizes the benchmark conclusions
- `assets/` contains charts and sample visuals copied from benchmark outputs

## Design Principles

- keep dataset handling separate from model evaluation
- report classification and segmentation separately
- prioritize recall-oriented model selection
- keep raw experiment history separate from the cleaned benchmark story
- keep optional deployment scaffolding outside the benchmark core

## Model Families

- ResNet-18 supervised classifier
- PaDiM anomaly detector
- PatchCore anomaly detector
- FastFlow anomaly detector
- WinCLIP zero/few-shot VLM
- anomaly + VLM score fusion

## Output Contract

Every claim in this repository is meant to map back to the benchmark outputs generated here. Hybrid segmentation and hybrid latency are not claimed where they were not fully measured.
