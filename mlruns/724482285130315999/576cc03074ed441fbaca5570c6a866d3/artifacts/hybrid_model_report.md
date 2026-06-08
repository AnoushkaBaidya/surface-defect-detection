# Hybrid Model Report

## Best Hybrid Configuration

| Model | Classification AUROC | F1 | Precision | Recall | FNR | FN Count |
|---|---:|---:|---:|---:|---:|---:|
| PatchCore + WinCLIP k=4 (0.7 / 0.3) | 0.9853 | 0.9773 | 0.9731 | 0.9823 | 0.0177 | 1.53 |

## Comparison to PatchCore

| Metric | PatchCore | Best Hybrid | Absolute Change |
|---|---:|---:|---:|
| Classification AUROC | 0.9823 | 0.9853 | +0.0030 |
| F1 | 0.9710 | 0.9773 | +0.0064 |
| Recall | 0.9723 | 0.9823 | +0.0099 |
| FNR | 0.0277 | 0.0177 | -0.0099 |

## Interpretation

Hybrid fusion helped only when the anomaly detector stayed in control. Heavier VLM weighting reduced quality, which is a strong signal that semantic prompting should be an auxiliary feature rather than the backbone of the detector.

## Limitation

Hybrid latency is not treated as settled because end-to-end timing was not fully captured in the same evaluation contract as the standalone models.
