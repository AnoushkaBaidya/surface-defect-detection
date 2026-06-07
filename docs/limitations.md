# Limitations

## Scope Limits

- The active benchmark is restricted to MVTec AD.
- Internal validation data is intentionally excluded from this repository.
- The repository does not claim domain-shift robustness beyond the measured benchmark setup.

## Measurement Limits

- Hybrid latency was not captured under the same end-to-end runtime contract as the standalone models.
- WinCLIP measurements were collected in a CPU-local setup and should not be generalized to all hardware profiles.
- Runtime and artifact comparisons are useful for relative positioning inside this repo, not as universal deployment claims.

## Implementation Limits

- The FastAPI runtime path is intentionally lightweight and currently supports the ResNet18 classification path only.
- Anomaly-model serving is not yet implemented as a unified runtime interface.
- CI and tests validate the repository core, not full GPU-dependent training workflows.

## Research Limits

- ResNet18 uses supervised labels borrowed from MVTec test defects to simulate limited defect supervision.
- Model selection is benchmark-driven and may change under different weighting schemes or threshold-selection strategies.
