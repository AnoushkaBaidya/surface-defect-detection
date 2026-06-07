# VLM Evaluation

## WinCLIP Aggregate Results

| Variant | Classification AUROC | F1 | Recall | FNR | Segmentation AUROC | Segmentation F1 | Mean Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| WinCLIP k=0 | 0.8812 | 0.9185 | 0.9475 | 0.0525 | 0.8236 | 0.2483 | 15119.9 |
| WinCLIP k=1 | 0.8976 | 0.9239 | 0.9597 | 0.0403 | 0.9057 | 0.3768 | 19553.3 |
| WinCLIP k=4 | 0.9249 | 0.9258 | 0.9342 | 0.0658 | 0.9121 | 0.3983 | 17231.8 |

## What Worked

- few-shot context improved VLM utility over zero-shot
- k=1 was the strongest recall-oriented WinCLIP configuration
- k=4 improved classification AUROC and segmentation quality

## What Limited VLM Use

- latency was orders of magnitude worse than anomaly baselines
- standalone VLM quality did not surpass PatchCore or FastFlow
- production positioning is weak in a CPU-local setup

## Conclusion

WinCLIP is valuable as a benchmarked semantic reference model. In this repository it is best interpreted as an auxiliary signal source for hybrid fusion, not as the main industrial detector.
