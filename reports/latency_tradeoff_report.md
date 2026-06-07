# Latency Tradeoff Report

## Runtime Comparison

| Model | Latency (ms) | Throughput (img/s) | Size (MB) | Recall | FNR |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 50.8 | 20.88 | 42.7 | 0.6905 | 0.3095 |
| PaDiM | 447.1 | 2.29 | 561.8 | 0.9484 | 0.0516 |
| FastFlow | 502.7 | 2.05 | 268.7 | 0.9606 | 0.0394 |
| PatchCore | 1193.8 | 0.86 | 781.2 | 0.9723 | 0.0277 |
| WinCLIP k=1 | 19553.3 | 0.05 | ~0.0 | 0.9597 | 0.0403 |

## Takeaways

- ResNet-18 is operationally cheap but quality-limited
- PatchCore is quality-optimal but expensive
- FastFlow is the most balanced anomaly runtime option
- WinCLIP is too slow for primary deployment use in the measured setup
- hybrid latency needs dedicated remeasurement before deployment claims
