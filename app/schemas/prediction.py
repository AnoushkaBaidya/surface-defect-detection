from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str


class LatencyBreakdown(BaseModel):
    preprocess_ms: float
    resnet_inference_ms: float
    efficientad_inference_ms: float | None = None
    postprocess_ms: float
    total_ms: float


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    prediction: str
    resnet18_defect_score: float
    resnet18_threshold: float
    efficientad_enabled: bool
    efficientad_score: float | None = None
    efficientad_threshold: float | None = None
    decision_reason: str
    latency: LatencyBreakdown
