# Limitations

## Scope Limits

- The active benchmark is restricted to MVTec AD.
- Internal validation data is intentionally excluded from this repository.
- The repository does not claim domain-shift robustness beyond the measured benchmark setup.

## Measurement Limits

- Hybrid latency in the runtime optimization report is derived from the sequential PatchCore + WinCLIP bottle-component timings rather than a single exported hybrid artifact.
- WinCLIP measurements were collected in a CPU-local setup and should not be generalized to all hardware profiles.
- Runtime and artifact comparisons are useful for relative positioning inside this repo, not as universal deployment claims.

## Implementation Limits

- The FastAPI runtime path is intentionally lightweight and currently supports the ResNet18 classification path only.
- Anomaly-model serving is not yet implemented as a unified runtime interface.
- ONNX export and INT8 quantization are implemented for the ResNet18 deployment path only; the hybrid winner remains a component-based serving policy.
- CI and tests validate the repository core, not full GPU-dependent training workflows.

## Research Limits

- ResNet18 uses supervised labels borrowed from MVTec test defects to simulate limited defect supervision.
- Model selection is benchmark-driven and may change under different weighting schemes or threshold-selection strategies.
