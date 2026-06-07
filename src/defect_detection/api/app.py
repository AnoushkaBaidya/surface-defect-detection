"""
Minimal FastAPI runtime for classification inference.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from defect_detection.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    HealthResponse,
)
from defect_detection.inference.classifier import ImageClassifierService

app = FastAPI(title="surface-defect-detection", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health endpoint."""
    return HealthResponse(status="ok")


@app.post("/v1/predict/classification", response_model=ClassificationResponse)
def predict_classification(request: ClassificationRequest) -> ClassificationResponse:
    """Run one image classification prediction."""
    image_path = Path(request.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    if request.checkpoint_path is not None and not Path(request.checkpoint_path).exists():
        raise HTTPException(
            status_code=404, detail=f"Checkpoint not found: {request.checkpoint_path}"
        )

    try:
        service = ImageClassifierService(
            model_name=request.model_name,
            checkpoint_path=request.checkpoint_path,
            pretrained=request.pretrained,
            device=request.device,
            image_size=request.image_size,
        )
        result = service.predict(image_path)
    except KeyError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return ClassificationResponse(
        model_name=result.model_name,
        predicted_label=result.predicted_label,
        predicted_index=result.predicted_index,
        defect_probability=result.defect_probability,
        normal_probability=result.normal_probability,
        image_path=result.image_path,
    )
