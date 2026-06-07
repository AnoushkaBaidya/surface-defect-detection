from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.common import (
    REPO_ROOT,
    build_onnx_session,
    build_resnet18_binary_classifier,
    confusion_metrics,
    load_rows,
    load_torch_checkpoint,
    load_yaml,
    preprocess_pil,
    save_predictions_csv,
    score_distribution_summary,
    softmax,
    write_json,
)


def predict_with_onnx(
    rows: list[dict[str, Any]], model_path: Path, input_size: int
) -> list[dict[str, Any]]:
    session = build_onnx_session(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction_rows: list[dict[str, Any]] = []

    from PIL import Image

    for row in rows:
        with Image.open(REPO_ROOT / row["relative_path"]) as image:
            tensor = preprocess_pil(image, input_size)
        logits = session.run([output_name], {input_name: np.expand_dims(tensor, axis=0)})[0]
        probability = float(softmax(logits[0])[1])
        prediction_rows.append({**row, "resnet18_score": probability})

    return prediction_rows


def predict_with_checkpoint(
    rows: list[dict[str, Any]], model_path: Path, input_size: int, device_name: str
) -> list[dict[str, Any]]:
    import torch

    device = torch.device(device_name)
    model = build_resnet18_binary_classifier().to(device)
    checkpoint = load_torch_checkpoint(model_path, device)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()

    prediction_rows: list[dict[str, Any]] = []
    from PIL import Image

    with torch.no_grad():
        for row in rows:
            with Image.open(REPO_ROOT / row["relative_path"]) as image:
                tensor = preprocess_pil(image, input_size)
            logits = model(torch.from_numpy(np.expand_dims(tensor, axis=0)).to(device))
            probability = float(torch.softmax(logits, dim=1)[0, 1].cpu().item())
            prediction_rows.append({**row, "resnet18_score": probability})

    return prediction_rows


def evaluate_prediction_rows(
    prediction_rows: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    y_true = np.array([int(row["label_id"]) for row in prediction_rows], dtype=int)
    y_score = np.array([float(row["resnet18_score"]) for row in prediction_rows], dtype=float)

    metrics = confusion_metrics(y_true=y_true, y_score=y_score, threshold=threshold)
    metrics["score_distribution"] = {
        "all": score_distribution_summary(list(y_score)),
        "normal": score_distribution_summary(
            [float(row["resnet18_score"]) for row in prediction_rows if row["label"] == "normal"]
        ),
        "defective": score_distribution_summary(
            [float(row["resnet18_score"]) for row in prediction_rows if row["label"] == "defective"]
        ),
    }
    return metrics


def write_markdown_summary(path: Path, metrics: dict[str, Any], model_path: Path) -> None:
    lines = [
        "# Retained Sample Evaluation",
        "",
        f"- Model artifact: `{model_path.relative_to(REPO_ROOT)}`",
        f"- Threshold: `{metrics['threshold']:.4f}`",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Precision: `{metrics['precision']:.4f}`",
        f"- Recall: `{metrics['recall']:.4f}`",
        f"- F1: `{metrics['f1']:.4f}`",
        f"- False positives: `{metrics['false_positive']}`",
        f"- False negatives: `{metrics['false_negative']}`",
        "",
        "## Score Monitoring Summary",
        "",
        "| Group | Mean | Std | P05 | P50 | P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for group, summary in metrics["score_distribution"].items():
        lines.append(
            f"| {group} | {summary['mean']:.4f} | {summary['std']:.4f} | {summary['p05']:.4f} "
            f"| {summary['p50']:.4f} | {summary['p95']:.4f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    config_path: str = "configs/pipeline/evaluate_resnet18_retained_sample.yaml",
    from_saved_predictions: bool = False,
) -> dict[str, Any]:
    config = load_yaml(REPO_ROOT / config_path)
    model_path = REPO_ROOT / config["model"]["artifact_path"]
    prediction_csv = REPO_ROOT / config["outputs"]["prediction_csv"]

    if from_saved_predictions and prediction_csv.exists():
        with prediction_csv.open("r", encoding="utf-8") as handle:
            prediction_rows = list(csv.DictReader(handle))
    else:
        split_rows = load_rows(REPO_ROOT / config["dataset"]["evaluation_split_csv"])
        if model_path.suffix == ".onnx":
            prediction_rows = predict_with_onnx(
                rows=split_rows,
                model_path=model_path,
                input_size=int(config["model"]["input_size"]),
            )
        else:
            prediction_rows = predict_with_checkpoint(
                rows=split_rows,
                model_path=model_path,
                input_size=int(config["model"]["input_size"]),
                device_name=str(config["model"]["device"]),
            )
        save_predictions_csv(prediction_csv, prediction_rows)

    threshold_summary_path = REPO_ROOT / config["outputs"]["threshold_summary_json"]
    if threshold_summary_path.exists():
        threshold_payload = json.loads(threshold_summary_path.read_text(encoding="utf-8"))
        threshold = float(threshold_payload["selected_threshold"])
    else:
        threshold = float(config["thresholding"]["default_threshold"])

    metrics = evaluate_prediction_rows(prediction_rows=prediction_rows, threshold=threshold)
    metrics["prediction_csv"] = str(prediction_csv.relative_to(REPO_ROOT)).replace("\\", "/")
    metrics["model_artifact"] = str(model_path.relative_to(REPO_ROOT)).replace("\\", "/")

    write_json(REPO_ROOT / config["outputs"]["metrics_json"], metrics)
    write_json(
        REPO_ROOT / config["outputs"]["score_monitoring_json"], metrics["score_distribution"]
    )
    write_markdown_summary(REPO_ROOT / config["outputs"]["markdown_summary"], metrics, model_path)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pipeline/evaluate_resnet18_retained_sample.yaml",
    )
    parser.add_argument(
        "--from-saved-predictions",
        action="store_true",
        help="Regenerate metrics and reports from an existing saved prediction CSV.",
    )
    args = parser.parse_args()

    metrics = run(args.config, args.from_saved_predictions)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
