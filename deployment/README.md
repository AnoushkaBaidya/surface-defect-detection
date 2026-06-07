# Deployment Notes

This directory is reserved for operational packaging that sits outside the benchmark core.

Current repository scope:

- benchmark execution on `MVTec AD`
- report generation from benchmark artifacts
- a lightweight FastAPI inference path for the ResNet18 classifier

Anything added here should remain tightly coupled to implemented runtime behavior rather than placeholder infrastructure.
