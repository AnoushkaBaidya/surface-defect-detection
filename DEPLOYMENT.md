# Deployment

## Runtime

The inference service is a FastAPI application served by Uvicorn. Models are loaded once at startup using ONNX Runtime CPU execution.

Endpoints:

- `GET /health`
- `GET /metrics`
- `POST /predict`

## Docker

Build locally:

```bash
docker build -t surface-defect-mlops-api .
```

Run locally:

```bash
docker run --rm -p 8000:8000 surface-defect-mlops-api
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"
```

## Cloud Run

Build and push:

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/surface-defect-mlops-api
```

Deploy:

```bash
gcloud run deploy surface-defect-mlops-api \
  --image gcr.io/PROJECT_ID/surface-defect-mlops-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 120 \
  --allow-unauthenticated
```

For non-demo services, use authenticated access instead of `--allow-unauthenticated`.

## Environment Assumptions

- Python 3.11
- CPU inference through ONNX Runtime
- Model artifacts available under `models/`
- Model policy config available at `configs/final_model_policy.yaml`
- EfficientAD calibration config available at `configs/efficientad_calibration.json`
- Input images sent as multipart uploads

## Cost-Control Settings

Example Cloud Run settings for a small service:

| Setting | Recommendation | Reason |
|---|---|---|
| `min-instances` | `0` | Avoid idle cost |
| `max-instances` | `3` | Prevent runaway spend |
| `concurrency` | `4` | Balance latency and CPU contention |
| `memory` | `2Gi` | Room for ONNX sessions and image preprocessing |
| `cpu` | `2` | Stable CPU inference latency |
| `timeout` | `120` seconds | Enough headroom for larger images or cold starts |

Cloud Run settings should be adjusted using measured p95 latency, target throughput, and cost limits.

## CI/CD

The repo includes GitHub Actions for:

- dependency installation
- Ruff linting
- Black formatting check
- Pytest test suite
- Docker build validation

These workflows validate that the API, drift logic, retraining trigger, and Docker image remain buildable before running the service.

## Observability

The API exposes Prometheus metrics at `/metrics` and writes structured JSON logs for:

- model loading
- prediction completion
- prediction failure
- latency breakdown
- decision reason

Cloud Run logs can be filtered by `request_id` for debugging individual predictions.
