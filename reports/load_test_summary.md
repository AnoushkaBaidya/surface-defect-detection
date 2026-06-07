# Load Test Summary

## Purpose

The load-test utilities validate the inference API shape. The goal is not to claim traffic at unrealistic scale. The goal is to verify that the containerized API can accept concurrent image requests, return structured predictions, and report latency/throughput metrics.

## Method

The load test sends asynchronous multipart image uploads to the `/predict` endpoint:

```bash
python load_tests/cloudrun_load_test.py \
  --service-url https://YOUR_CLOUD_RUN_URL \
  --image-dir path/to/non-sensitive/load-test-images \
  --requests 1000 \
  --concurrency 4
```

For local testing, use synthetic or non-sensitive images:

```bash
python load_tests/generate_synthetic_load_images.py \
  --source-dir path/to/source-images \
  --output-dir load_tests/generated \
  --count 1000
```

`load_tests/generated/` is ignored by Git.

## Reported Metrics

The load-test script reports:

- total requests
- success count
- error count
- success rate
- total runtime
- requests per second
- estimated requests per day
- mean latency
- p50 latency
- p95 latency
- max latency

## Interpretation

The load test should be used as an engineering smoke test before running the service on Cloud Run. A passing load test means the API can serve concurrent requests with acceptable latency in the tested environment. It does not prove future capacity by itself; final sizing should be based on expected line speed, image size, Cloud Run CPU/memory settings, and p95 latency.

## Production Constraint

The model policy intentionally uses a selective anomaly gate: EfficientAD runs only when ResNet18 does not already flag the image. This reduces unnecessary second-stage inference and keeps latency closer to the supervised baseline on obvious defects.
