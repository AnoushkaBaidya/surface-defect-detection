# Runtime Tradeoffs

## Objective

Runtime comparison in this repository is used to position model families relative to one another, not to make blanket deployment claims.

## Relative Runtime Signals

- ResNet-18 is the lightest supervised baseline.
- FastFlow provides a stronger latency/model-size profile than PatchCore.
- PatchCore remains the strongest pure anomaly model, but it is the most expensive anomaly baseline in runtime terms.
- WinCLIP adds semantic signal at a large runtime cost.

## Ranking Philosophy

Runtime-aware ranking in this repository combines:

- recall
- false-negative rate
- segmentation quality where available
- mean inference latency
- model artifact size

This weighting is encoded in the benchmark summary pipeline and is intended to support comparative reasoning rather than universal ranking.

## Current Runtime Surface

- benchmark scripts measure inference latency and throughput where supported
- MLflow captures runtime metrics alongside quality metrics
- a minimal FastAPI runtime path is available for the ResNet18 classification path

## Remaining Gaps

- no unified anomaly-model runtime interface
- no batch-serving benchmark harness
- no GPU/CPU comparison matrix
- no production traffic simulation
