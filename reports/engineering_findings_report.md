# Engineering Findings Report

## Key Findings

1. Supervised classification alone is not reliable enough for open-set industrial inspection.
2. Normal-only anomaly detection substantially reduces missed defects.
3. PatchCore is the quality leader, but not the easiest model to deploy.
4. FastFlow is the best compromise when deployment feasibility matters.
5. VLMs add useful semantic signal, but their runtime cost is severe.
6. Hybrid fusion helps only when it remains anomaly-dominant.

## Safe Conclusions

- anomaly detection is the correct center of gravity for this benchmark
- recall and FNR are the right decision metrics for final selection
- the repo supports a nuanced recommendation rather than a single universal winner

## Recommendation

Use PatchCore as the pure-quality benchmark reference, FastFlow as the deployment-oriented anomaly candidate, and the PatchCore + WinCLIP hybrid as the research-quality aggregate winner.
