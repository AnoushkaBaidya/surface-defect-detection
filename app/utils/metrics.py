from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "defect_api_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "defect_predictions_total",
    "Total predictions by label",
    ["prediction"],
)

LATENCY_HISTOGRAM = Histogram(
    "defect_prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
