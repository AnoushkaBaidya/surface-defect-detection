from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_to_int(label: str) -> int:
    return 1 if label == "defective" else 0


def label_to_name(label: int) -> str:
    return "defective" if label == 1 else "normal"


def build_resnet18_binary_classifier() -> Any:
    import torch.nn as nn
    import torchvision.models as models

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def preprocess_pil(image: Image.Image, input_size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((input_size, input_size))
    array = np.asarray(resized).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(array, dtype=np.float32)


def make_loader(
    rows: list[dict[str, Any]],
    input_size: int,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    class SurfaceDefectDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]], input_size: int) -> None:
            self.rows = rows
            self.input_size = input_size

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            row = self.rows[index]
            with Image.open(REPO_ROOT / row["relative_path"]) as image:
                tensor = preprocess_pil(image, self.input_size)
            return torch.from_numpy(tensor), int(row["label_id"])

    dataset = SurfaceDefectDataset(rows=rows, input_size=input_size)

    if use_weighted_sampler:
        label_counts: dict[int, int] = {}
        for row in rows:
            label_counts[row["label_id"]] = label_counts.get(row["label_id"], 0) + 1
        weights = [1.0 / label_counts[row["label_id"]] for row in rows]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def confusion_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def score_distribution_summary(values: list[float]) -> dict[str, float]:
    array = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def load_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_predictions_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    preferred_order = [
        "dataset_name",
        "capture_batch",
        "label",
        "label_id",
        "resnet18_score",
    ]
    fieldnames = [name for name in preferred_order if any(name in row for row in rows)]
    redacted_rows = [{key: row[key] for key in fieldnames} for row in rows]
    write_rows(path, redacted_rows, fieldnames)


def build_onnx_session(path: str | Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass(frozen=True)
class TorchCheckpoint:
    model_state_dict: dict[str, Any]
    metadata: dict[str, Any]


def load_torch_checkpoint(path: str | Path, device: Any) -> TorchCheckpoint:
    import torch

    payload = torch.load(path, map_location=device)

    if "model_state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint format at {path}")

    metadata = payload.get("metadata", {})
    return TorchCheckpoint(model_state_dict=payload["model_state_dict"], metadata=metadata)
