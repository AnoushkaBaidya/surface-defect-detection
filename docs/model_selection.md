# Model Selection

## Selection Objective

The model-selection process in this repository prioritizes missed-defect reduction over headline accuracy.

## Selection Criteria

Primary criteria:

1. recall
2. false-negative rate
3. classification AUROC and F1
4. segmentation quality where available

Secondary criteria:

1. mean inference latency
2. throughput
3. model artifact size

## Final Positioning

- Best supervised baseline: ResNet-18
- Best pure anomaly detector: PatchCore
- Best anomaly runtime/size tradeoff: FastFlow
- Best VLM benchmark: WinCLIP few-shot variants
- Best aggregate benchmark result: PatchCore + WinCLIP k=4 with 0.7 anomaly / 0.3 VLM weighting

## Interpretation

The selected hybrid result should be read as the strongest benchmark outcome in this repository, not as a universally validated operational choice. The hybrid gain is real in the measured benchmark outputs, but latency validation is not yet complete under the same runtime contract as the standalone models.
