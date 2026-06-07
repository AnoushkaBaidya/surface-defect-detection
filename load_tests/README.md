# Load Tests

This folder contains lightweight API load-test utilities for the inference service.

## Generate Safe Test Images

Use only synthetic, sample, or otherwise non-sensitive images.

```bash
python load_tests/generate_synthetic_load_images.py \
  --source-dir path/to/source-images \
  --output-dir load_tests/generated \
  --count 1000
```

`load_tests/generated/` is ignored by Git.

## Run API Load Test

```bash
python load_tests/cloudrun_load_test.py \
  --service-url http://localhost:8000 \
  --image-dir load_tests/generated \
  --requests 1000 \
  --concurrency 4
```

For Cloud Run, replace `--service-url` with the service URL.

## Metrics

The script prints:

- success rate
- requests per second
- estimated requests per day
- mean latency
- p50 latency
- p95 latency
- max latency

These numbers are for service sizing and regression checks. They should not be oversold as guaranteed throughput.
