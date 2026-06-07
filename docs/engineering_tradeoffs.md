# Engineering Tradeoffs

## Quality vs Deployment Cost

| Model | Recall | FNR | Mean Latency (ms) | Throughput (img/s) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 0.6905 | 0.3095 | 50.8 | 20.88 | 42.7 |
| PaDiM | 0.9484 | 0.0516 | 447.1 | 2.29 | 561.8 |
| PatchCore | 0.9723 | 0.0277 | 1193.8 | 0.86 | 781.2 |
| FastFlow | 0.9606 | 0.0394 | 502.7 | 2.05 | 268.7 |
| WinCLIP k=1 | 0.9597 | 0.0403 | 19553.3 | 0.05 | ~0.0 |

## Interpretation

- ResNet-18 is fastest, but misses too many defects
- PatchCore is best on quality, but most expensive among anomaly baselines
- FastFlow offers the most balanced anomaly deployment profile
- WinCLIP is too slow to justify standalone use here

## Recommended Positioning

- best aggregate result: PatchCore + WinCLIP hybrid
- pure anomaly winner: PatchCore
- best anomaly runtime/size tradeoff: FastFlow
