# Benchmark Design Decisions

## Why Compare Supervised and Anomaly Models

A supervised CNN is a useful baseline, but industrial inspection rarely has complete defect coverage. The benchmark therefore compares supervised classification against normal-only anomaly detection rather than assuming one family is sufficient.

## Why Recall and FNR Are Central

Missed defects are usually more costly than extra review burden. This is why model selection in this repository is not driven by AUROC alone.

## Why WinCLIP Was Included

The VLM benchmark tests whether semantic priors add value under limited defect supervision. It is an evaluation branch, not a marketing addition.

## Why Hybrid Fusion Is Conservative

The best hybrid result came from anomaly-dominant weighting. This supports a disciplined design choice: use semantic models to complement, not replace, anomaly evidence.

## Honest Deferrals

- EfficientAD was deferred because CPU-local training cost was impractical
- AnomalyCLIP remained reference-only because reproduction cost was not justified in the current setup
- internal-domain validation is intentionally excluded from this repo narrative
