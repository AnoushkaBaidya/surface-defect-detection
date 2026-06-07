# Cloud Run Deployment Notes

This folder documents the intended Cloud Run target for the FastAPI inference container.

Recommended demo settings:

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
  --timeout 120
```

Use authenticated access for non-demo services.
