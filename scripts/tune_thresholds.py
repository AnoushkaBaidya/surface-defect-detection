from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.common import REPO_ROOT, confusion_metrics, load_yaml, write_json, write_rows


def load_prediction_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_score: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            y_true.append(int(row["label_id"]))
            y_score.append(float(row["resnet18_score"]))
    return np.array(y_true, dtype=int), np.array(y_score, dtype=float)


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    recall_floor: float,
) -> tuple[dict, list[dict]]:
    best_metrics: dict | None = None
    sweep_rows: list[dict] = []

    for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
        metrics = confusion_metrics(y_true=y_true, y_score=y_score, threshold=float(threshold))
        sweep_rows.append(metrics)

        meets_recall_floor = metrics["recall"] >= recall_floor
        if best_metrics is None:
            best_metrics = metrics
            continue

        best_meets_recall_floor = best_metrics["recall"] >= recall_floor
        if meets_recall_floor and not best_meets_recall_floor:
            best_metrics = metrics
        elif meets_recall_floor == best_meets_recall_floor and metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Threshold tuning did not evaluate any thresholds.")

    return best_metrics, sweep_rows


def run(
    config_path: str = "configs/pipeline/evaluate_resnet18_retained_sample.yaml",
    predictions_override: str | None = None,
) -> dict:
    config = load_yaml(REPO_ROOT / config_path)
    predictions_path = (
        REPO_ROOT / predictions_override
        if predictions_override
        else REPO_ROOT / config["outputs"]["prediction_csv"]
    )

    y_true, y_score = load_prediction_arrays(predictions_path)
    best_metrics, sweep_rows = select_threshold(
        y_true=y_true,
        y_score=y_score,
        threshold_min=float(config["thresholding"]["sweep_min"]),
        threshold_max=float(config["thresholding"]["sweep_max"]),
        threshold_steps=int(config["thresholding"]["sweep_steps"]),
        recall_floor=float(config["thresholding"]["recall_floor"]),
    )

    threshold_payload = {
        "selected_threshold": best_metrics["threshold"],
        "selection_objective": "maximize_f1_subject_to_recall_floor",
        "recall_floor": float(config["thresholding"]["recall_floor"]),
        "selected_metrics": best_metrics,
        "prediction_csv": str(predictions_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    write_json(REPO_ROOT / config["outputs"]["threshold_summary_json"], threshold_payload)
    write_rows(
        REPO_ROOT / config["outputs"]["threshold_sweep_csv"],
        sweep_rows,
        list(sweep_rows[0].keys()),
    )
    return threshold_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pipeline/evaluate_resnet18_retained_sample.yaml",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Override the prediction CSV used for threshold tuning.",
    )
    args = parser.parse_args()

    threshold_payload = run(args.config, args.predictions)

    print(json.dumps(threshold_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
