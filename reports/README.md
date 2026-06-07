# Reports

This directory summarizes the model-selection work for the surface-defect inspection system. The reports are ordered as a decision narrative:

1. [Model Selection Report](model_selection_report.md)
2. [ResNet18 Failure Analysis](resnet18_failure_analysis.md)
3. [Anomaly Model Analysis](anomaly_model_analysis.md)
4. [Final Hybrid Policy Report](final_hybrid_policy_report.md)
5. [Load Test Summary](load_test_summary.md)

The analysis uses the internal Oct14-to-Nov8 transfer setting: Oct14 is the source/training domain and Nov8 is the held-out validation domain. Nov8 labels support classification evaluation only; pixel masks were not available for segmentation metrics.

Figures are kept under [figures](figures/) and linked from the reports where they support the result being discussed. Raw inspection images are intentionally excluded.
