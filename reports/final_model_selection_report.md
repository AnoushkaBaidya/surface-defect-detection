# Final Model Selection Report

## Selected Model

**PatchCore + WinCLIP k=4 with 0.7 anomaly / 0.3 VLM weighting**

## Selection Metrics

| Classification AUROC | F1 | Precision | Recall | FNR | Mean FN Count |
|---:|---:|---:|---:|---:|---:|
| 0.9853 | 0.9773 | 0.9731 | 0.9823 | 0.0177 | 1.53 |

## Why It Was Selected

- strongest aggregate balance of recall, FNR, AUROC, and F1
- improves on PatchCore without handing control to the VLM
- supports the conclusion that semantic signals are most useful as small auxiliary terms

## Model Positioning

| Role | Model | Rationale |
|---|---|---|
| Best supervised baseline | ResNet-18 | fast and simple, but weak on missed defects |
| Best pure anomaly detector | PatchCore | strongest standalone quality |
| Best anomaly runtime/size tradeoff | FastFlow | better quality-to-runtime tradeoff |
| Best VLM benchmark | WinCLIP | useful semantic signal, high latency cost |
| Best overall quality | PatchCore + WinCLIP hybrid | strongest aggregate recall/FNR balance |

## Important Limitation

Hybrid latency was not fully measured end-to-end, so this should be treated as the strongest benchmark result rather than a final runtime recommendation.
