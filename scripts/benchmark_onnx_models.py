from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from scripts.common import REPO_ROOT, build_onnx_session, load_yaml


def benchmark(session_path: str, input_size: int, output_name: str, runs: int) -> dict:
    session = build_onnx_session(REPO_ROOT / session_path)
    input_name = session.get_inputs()[0].name
    if output_name == "auto":
        output_name = session.get_outputs()[0].name

    latencies_ms: list[float] = []
    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    for index in range(runs + 10):
        start = time.perf_counter()
        session.run([output_name], {input_name: dummy_input})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if index >= 10:
            latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    return {
        "runs": runs,
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": latencies_ms[int(0.50 * (len(latencies_ms) - 1))],
        "p95_ms": latencies_ms[int(0.95 * (len(latencies_ms) - 1))],
        "p99_ms": latencies_ms[int(0.99 * (len(latencies_ms) - 1))],
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pipeline/evaluate_resnet18_retained_sample.yaml",
    )
    args = parser.parse_args()

    config = load_yaml(REPO_ROOT / args.config)
    payload = [
        {
            "model": "resnet18_onnx",
            **benchmark(
                session_path=config["model"]["artifact_path"],
                input_size=int(config["model"]["input_size"]),
                output_name="auto",
                runs=100,
            ),
        },
        {
            "model": "efficientad_onnx",
            **benchmark(
                session_path="models/efficientad_score.onnx",
                input_size=224,
                output_name="auto",
                runs=50,
            ),
        },
    ]

    output_path = REPO_ROOT / "benchmark_outputs/onnx_runtime_latency_benchmark.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
