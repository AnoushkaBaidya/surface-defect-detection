# Offline and Online Architecture

```mermaid
flowchart LR
    subgraph Offline[Offline ML Pipeline]
        A[local retained images or internal dataset] --> B[scripts.build_dataset_manifest]
        B --> C[scripts.create_retained_sample_split]
        C --> D[scripts.train_resnet18]
        D --> E[scripts.export_onnx]
        C --> F[scripts.evaluate_resnet18]
        F --> G[scripts.tune_thresholds]
        F --> H[score monitoring summary]
        G --> I[threshold provenance JSON]
        E --> J[artifact lineage JSON]
        H --> K[reports/results_manifest.json]
        I --> K
        J --> K
    end

    subgraph Online[Serving and Monitoring]
        L[client image upload] --> M[FastAPI /predict]
        M --> N[dimension and payload guardrails]
        N --> O[ResNet18 ONNX]
        O --> P{score >= threshold?}
        P -- yes --> Q[defective]
        P -- no --> R[EfficientAD ONNX gate]
        R --> S{normalized score >= 0.30?}
        S -- yes --> Q
        S -- no --> T[normal]
        Q --> U[JSON response + logs + Prometheus metrics]
        T --> U
        U --> V[drift baseline / score monitoring / retraining manifest]
    end
```

## Notes

- The repo has an inspectable offline pipeline, even though the full data workflow is intentionally excluded.
- `scripts.reproduce_retained_sample_pipeline --from-saved-predictions` regenerates report artifacts from tracked retained-sample predictions.
- `reports/results_manifest.json` ties the visible reports back to exact artifacts.
