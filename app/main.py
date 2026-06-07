from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import AppConfig, load_app_config, resolve_repo_path
from app.schemas.prediction import HealthResponse, LatencyBreakdown, PredictionResponse
from app.services.efficientad_onnx_service import EfficientADOnnxService
from app.services.preprocessing import (
    load_image,
    preprocess_pil_for_efficientad,
    preprocess_pil_for_resnet,
    validate_image_dimensions,
)
from app.services.resnet18_onnx_service import ResNet18OnnxService
from app.utils.logging import log_json
from app.utils.metrics import LATENCY_HISTOGRAM, PREDICTION_COUNT, REQUEST_COUNT

config: AppConfig | None = None
resnet_service: ResNet18OnnxService | None = None
efficientad_service: EfficientADOnnxService | None = None

PREDICT_FILE = File(...)


def load_models() -> None:
    global config, resnet_service, efficientad_service

    config = load_app_config()
    onnx_path = resolve_repo_path(config.resnet18.onnx_path)

    resnet_service = ResNet18OnnxService(onnx_path=onnx_path)

    if (
        config.efficientad.enabled
        and config.efficientad.runtime == "onnx"
        and config.efficientad.onnx_path is not None
        and config.efficientad.calibration_path is not None
    ):
        efficientad_service = EfficientADOnnxService(
            onnx_path=resolve_repo_path(config.efficientad.onnx_path),
            calibration_path=resolve_repo_path(config.efficientad.calibration_path),
        )

    log_json(
        "models_loaded",
        {
            "model_name": config.model_policy.name,
            "model_version": config.model_policy.version,
            "resnet_onnx_path": str(onnx_path),
            "efficientad_enabled": config.efficientad.enabled,
            "efficientad_loaded": efficientad_service is not None,
            "efficientad_runtime": config.efficientad.runtime,
        },
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_models()
    yield


app = FastAPI(
    title="Surface Defect Detection API",
    version="1.0.0",
    description="Production-style API for ResNet18 + anomaly safety-gate defect prediction.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if config is None:
        raise HTTPException(status_code=503, detail="Config not loaded")

    return HealthResponse(
        status="ok",
        model_loaded=resnet_service is not None,
        model_name=config.model_policy.name,
        model_version=config.model_policy.version,
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = PREDICT_FILE) -> PredictionResponse:
    if config is None or resnet_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    start_total = time.perf_counter()

    if file.content_type is not None and not file.content_type.startswith("image/"):
        REQUEST_COUNT.labels(endpoint="/predict", status="bad_request").inc()
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    if len(image_bytes) > config.api_limits.max_upload_bytes:
        REQUEST_COUNT.labels(endpoint="/predict", status="bad_request").inc()
        raise HTTPException(status_code=413, detail="Uploaded image exceeds configured size limit")

    try:
        preprocess_start = time.perf_counter()
        pil_image = load_image(image_bytes)
        validate_image_dimensions(
            image=pil_image,
            min_dimension=config.api_limits.min_image_dimension,
            max_dimension=config.api_limits.max_image_dimension,
        )

        image_tensor = preprocess_pil_for_resnet(
            image=pil_image,
            input_size=config.resnet18.input_size,
        )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        resnet_start = time.perf_counter()
        resnet_score = resnet_service.predict_score(image_tensor)
        resnet_inference_ms = (time.perf_counter() - resnet_start) * 1000.0

        resnet_defective = resnet_score >= config.resnet18.raw_threshold

        # EfficientAD ONNX runs only when ResNet18 does not already flag the image.
        # This preserves the calibrated high-confidence safety-gate policy.
        efficientad_raw_score = None
        efficientad_score = None
        efficientad_defective = False
        efficientad_inference_ms = None

        if not resnet_defective and config.efficientad.enabled and efficientad_service is not None:
            efficientad_tensor = preprocess_pil_for_efficientad(
                image=pil_image,
                input_size=config.efficientad.input_size,
            )

            efficientad_start = time.perf_counter()
            efficientad_scores = efficientad_service.predict_scores(efficientad_tensor)
            efficientad_inference_ms = (time.perf_counter() - efficientad_start) * 1000.0

            efficientad_raw_score = efficientad_scores["raw_score"]
            efficientad_score = efficientad_scores["normalized_score"]

            efficientad_defective = efficientad_score >= config.efficientad.threshold

        postprocess_start = time.perf_counter()

        if resnet_defective:
            prediction = "defective"
            decision_reason = "resnet18_score_above_threshold"
        elif efficientad_defective:
            prediction = "defective"
            decision_reason = "efficientad_high_confidence_gate"
        else:
            prediction = "normal"
            decision_reason = "all_scores_below_threshold"

        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        total_ms = (time.perf_counter() - start_total) * 1000.0

        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        PREDICTION_COUNT.labels(prediction=prediction).inc()
        LATENCY_HISTOGRAM.observe(total_ms / 1000.0)

        response = PredictionResponse(
            request_id=request_id,
            model_name=config.model_policy.name,
            model_version=config.model_policy.version,
            prediction=prediction,
            resnet18_defect_score=resnet_score,
            resnet18_threshold=config.resnet18.raw_threshold,
            efficientad_enabled=config.efficientad.enabled,
            efficientad_score=efficientad_score,
            efficientad_threshold=config.efficientad.threshold,
            decision_reason=decision_reason,
            latency=LatencyBreakdown(
                preprocess_ms=preprocess_ms,
                resnet_inference_ms=resnet_inference_ms,
                efficientad_inference_ms=efficientad_inference_ms,
                postprocess_ms=postprocess_ms,
                total_ms=total_ms,
            ),
        )

        log_json(
            "prediction_completed",
            {
                "request_id": request_id,
                "filename": file.filename,
                "prediction": prediction,
                "resnet18_defect_score": resnet_score,
                "resnet18_threshold": config.resnet18.raw_threshold,
                "efficientad_raw_score": efficientad_raw_score,
                "efficientad_normalized_score": efficientad_score,
                "latency_ms": total_ms,
                "preprocess_ms": preprocess_ms,
                "resnet_inference_ms": resnet_inference_ms,
                "efficientad_inference_ms": efficientad_inference_ms,
                "postprocess_ms": postprocess_ms,
                "decision_reason": decision_reason,
            },
        )

        return response

    except ValueError as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="bad_request").inc()

        log_json(
            "prediction_bad_request",
            {
                "request_id": request_id,
                "filename": file.filename,
                "error": str(exc),
            },
        )

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()

        log_json(
            "prediction_failed",
            {
                "request_id": request_id,
                "filename": file.filename,
                "error": str(exc),
            },
        )

        raise HTTPException(status_code=500, detail="Prediction failed") from exc
