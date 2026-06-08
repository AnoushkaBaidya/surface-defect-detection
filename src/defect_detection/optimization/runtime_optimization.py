"""
Helpers for runtime benchmarking, ONNX export, and INT8 quantization.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

from defect_detection.runtime.predictor import build_classification_transform


@dataclass(frozen=True)
class LatencySummary:
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_images_per_second: float


@dataclass(frozen=True)
class ClassificationSummary:
    precision: float
    recall: float
    f1: float
    false_negative_rate: float


def load_json(path: str | Path) -> dict:
    """Load a JSON file into a dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: dict, path: str | Path) -> None:
    """Persist a JSON payload to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_bottle_eval_split(project_root: str | Path) -> pd.DataFrame:
    """Load the tracked bottle evaluation split."""
    project_root = Path(project_root)
    return pd.read_csv(
        project_root / "data/processed/phase3/resnet18/splits/bottle/eval.csv",
    )


def normalize_scores(values: list[float] | np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D score vector safely."""
    array = np.asarray(values, dtype=np.float64)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum == minimum:
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def fuse_scores(
    anomaly_scores: list[float] | np.ndarray,
    vlm_scores: list[float] | np.ndarray,
    *,
    anomaly_weight: float = 0.7,
    vlm_weight: float = 0.3,
) -> np.ndarray:
    """Apply the benchmark hybrid score-fusion rule."""
    anomaly_norm = normalize_scores(anomaly_scores)
    vlm_norm = normalize_scores(vlm_scores)
    return (anomaly_weight * anomaly_norm) + (vlm_weight * vlm_norm)


def summarize_latencies(latencies_seconds: list[float]) -> LatencySummary:
    """Convert per-image latencies to mean/p95/throughput summary."""
    if not latencies_seconds:
        return LatencySummary(0.0, 0.0, 0.0)

    latencies_ms = np.asarray(latencies_seconds, dtype=np.float64) * 1000.0
    total_seconds = float(np.sum(latencies_seconds))
    throughput = float(len(latencies_seconds) / total_seconds) if total_seconds > 0 else 0.0
    return LatencySummary(
        mean_latency_ms=float(np.mean(latencies_ms)),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        throughput_images_per_second=throughput,
    )


def summarize_binary_classification(
    labels: list[int] | np.ndarray,
    defect_scores: list[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> ClassificationSummary:
    """Compute compact binary metrics from defect scores."""
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(defect_scores, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)

    false_negative_count = int(np.sum((y_true == 1) & (y_pred == 0)))
    positive_count = int(np.sum(y_true == 1))
    false_negative_rate = float(false_negative_count / positive_count) if positive_count else 0.0

    return ClassificationSummary(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        false_negative_rate=false_negative_rate,
    )


def load_resnet_checkpoint(checkpoint_path: str | Path, *, device: str = "cpu") -> dict:
    """Load the public ResNet-18 checkpoint payload."""
    return torch.load(Path(checkpoint_path), map_location=device)


def prepare_image_tensor(image_path: str | Path, *, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess one image for ResNet-18 / ONNX inference."""
    image = Image.open(image_path).convert("RGB")
    transform = build_classification_transform(image_size)
    return transform(image).unsqueeze(0)


def benchmark_pytorch_classifier(
    model: torch.nn.Module,
    image_paths: list[Path],
    labels: list[int],
    *,
    image_size: int = 224,
    device: str = "cpu",
) -> dict:
    """Benchmark a PyTorch classification model on single-image inference."""
    model.eval()
    latencies: list[float] = []
    defect_scores: list[float] = []

    with torch.no_grad():
        for image_path in image_paths:
            tensor = prepare_image_tensor(image_path, image_size=image_size).to(device)
            start = time.perf_counter()
            logits = model(tensor)
            elapsed = time.perf_counter() - start
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

            latencies.append(elapsed)
            defect_scores.append(float(probabilities[1]))

    latency_summary = summarize_latencies(latencies)
    classification_summary = summarize_binary_classification(labels, defect_scores)

    return {
        "latency": asdict(latency_summary),
        "classification": asdict(classification_summary),
        "defect_scores": defect_scores,
    }


def export_resnet18_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    *,
    image_size: int = 224,
) -> Path:
    """Export a ResNet-18 binary classifier to ONNX."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy_input,
        destination,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    return destination


def build_onnx_session(onnx_path: str | Path) -> ort.InferenceSession:
    """Create a CPU ONNX Runtime session."""
    session_options = ort.SessionOptions()
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def benchmark_onnx_classifier(
    session: ort.InferenceSession,
    image_paths: list[Path],
    labels: list[int],
    *,
    image_size: int = 224,
) -> dict:
    """Benchmark an ONNX classifier on single-image inference."""
    latencies: list[float] = []
    defect_scores: list[float] = []
    input_name = session.get_inputs()[0].name

    for image_path in image_paths:
        tensor = prepare_image_tensor(image_path, image_size=image_size).numpy()
        start = time.perf_counter()
        logits = session.run(None, {input_name: tensor})[0]
        elapsed = time.perf_counter() - start

        logits_tensor = torch.from_numpy(logits)
        probabilities = torch.softmax(logits_tensor, dim=1)[0].numpy()

        latencies.append(elapsed)
        defect_scores.append(float(probabilities[1]))

    latency_summary = summarize_latencies(latencies)
    classification_summary = summarize_binary_classification(labels, defect_scores)

    return {
        "latency": asdict(latency_summary),
        "classification": asdict(classification_summary),
        "defect_scores": defect_scores,
    }


def model_size_mb(path: str | Path) -> float:
    """Return file size in megabytes."""
    file_path = Path(path)
    return float(file_path.stat().st_size / (1024 * 1024))


def max_probability_delta(
    reference_scores: list[float],
    candidate_scores: list[float],
) -> float:
    """Return the max absolute difference between two score vectors."""
    reference = np.asarray(reference_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    return float(np.max(np.abs(reference - candidate)))


def mean_probability_delta(
    reference_scores: list[float],
    candidate_scores: list[float],
) -> float:
    """Return the mean absolute difference between two score vectors."""
    reference = np.asarray(reference_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    return float(np.mean(np.abs(reference - candidate)))
