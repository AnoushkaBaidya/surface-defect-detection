# WinCLIP Evaluation Report

## Aggregate Metrics

| Variant | Classification AUROC | F1 | Recall | FNR | Segmentation AUROC | Segmentation F1 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| k=0 | 0.8812 | 0.9185 | 0.9475 | 0.0525 | 0.8236 | 0.2483 | 15119.9 |
| k=1 | 0.8976 | 0.9239 | 0.9597 | 0.0403 | 0.9057 | 0.3768 | 19553.3 |
| k=4 | 0.9249 | 0.9258 | 0.9342 | 0.0658 | 0.9121 | 0.3983 | 17231.8 |

## Findings

- k=1 produced the best recall and FNR among WinCLIP variants
- k=4 improved AUROC and segmentation quality
- all VLM variants were far slower than the anomaly detectors

## Conclusion

WinCLIP is valuable as a semantic benchmark branch, but not as the final standalone detector in this CPU-local setting.
