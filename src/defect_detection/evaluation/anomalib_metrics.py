"""
Utilities for extracting standardized metrics from Anomalib outputs.

Why this file exists
--------------------
Anomalib already reports image-level and pixel-level metrics such as AUROC/F1.
However, industrial inspection also requires:
- recall
- false-negative rate
- false-negative count
- latency
- throughput

This file creates a bridge between Anomalib outputs and the standardized
BenchmarkResult schema used across the benchmark suite.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def normalize_anomalib_test_results(test_results: Any) -> dict[str, float | None]:
    """
    Normalize Anomalib engine.test output into the benchmark metric names.

    Parameters
    ----------
    test_results:
        Output returned by Anomalib Engine.test.

    Returns
    -------
    dict[str, float | None]
        Standardized metric dictionary.
    """
    if isinstance(test_results, list) and test_results:
        raw = test_results[0]
    elif isinstance(test_results, dict):
        raw = test_results
    else:
        raw = {}

    def get_float(*keys: str) -> float | None:
        for key in keys:
            if key in raw:
                try:
                    return float(raw[key])
                except (TypeError, ValueError):
                    return None
        return None

    return {
        "classification_auroc": get_float("image_AUROC", "image_auroc", "classification_auroc"),
        "classification_f1": get_float("image_F1Score", "image_F1", "image_f1"),
        "segmentation_auroc": get_float("pixel_AUROC", "pixel_auroc", "segmentation_auroc"),
        "segmentation_f1": get_float("pixel_F1Score", "pixel_F1", "pixel_f1"),
        "segmentation_aupro": get_float("pixel_AUPRO", "aupro", "segmentation_aupro"),
    }


def extract_field_from_prediction(prediction: Any, field_names: list[str]) -> Any:
    """
    Extract a field from an Anomalib prediction object or dictionary.

    Anomalib prediction outputs may vary by version. Some versions return dicts;
    others return batch-like objects with attributes. This helper supports both.

    Parameters
    ----------
    prediction:
        Prediction object from Engine.predict.
    field_names:
        Candidate field names to try.

    Returns
    -------
    Any
        Extracted field value or None.
    """
    for field_name in field_names:
        if isinstance(prediction, dict) and field_name in prediction:
            return prediction[field_name]

        if hasattr(prediction, field_name):
            return getattr(prediction, field_name)

    return None


def to_numpy_array(value: Any) -> np.ndarray | None:
    """
    Convert tensors/lists/scalars to a NumPy array.

    Parameters
    ----------
    value:
        Input value.

    Returns
    -------
    np.ndarray | None
        Converted array or None.
    """
    if value is None:
        return None

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()

    array = np.asarray(value)
    return array.reshape(-1)


def compute_confusion_metrics_from_predictions(
    predictions: list[Any],
) -> dict[str, float | int | None]:
    """
    Compute recall/FNR from Anomalib predictions if labels are available.

    Expected fields vary by Anomalib version. We try common names:
    - ground truth label: gt_label, gt_labels, label, labels
    - prediction label: pred_label, pred_labels

    Returns
    -------
    dict[str, float | int | None]
        Confusion-style metrics, or None values if unavailable.
    """
    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []

    for prediction in predictions:
        true_value = extract_field_from_prediction(
            prediction,
            ["gt_label", "gt_labels", "label", "labels"],
        )
        pred_value = extract_field_from_prediction(
            prediction,
            ["pred_label", "pred_labels", "prediction", "predictions"],
        )

        true_array = to_numpy_array(true_value)
        pred_array = to_numpy_array(pred_value)

        if true_array is None or pred_array is None:
            continue

        y_true_batches.append(true_array.astype(int))
        y_pred_batches.append(pred_array.astype(int))

    if not y_true_batches or not y_pred_batches:
        return {
            "precision": None,
            "recall": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
            "false_positive_count": None,
            "false_negative_count": None,
        }

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)

    # MVTec convention:
    # 0 = normal
    # 1 = anomalous/defective
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "classification_f1_from_predictions": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
    }


def extract_prediction_scores_table(predictions: list[Any]) -> list[dict[str, Any]]:
    """
    Extract per-image scores, labels, and paths from Anomalib predictions.

    Returns
    -------
    list[dict[str, Any]]
        Rows containing image_path, label, pred_score, and optional pred_label.
    """
    rows: list[dict[str, Any]] = []

    for prediction in predictions:
        image_paths = extract_field_from_prediction(prediction, ["image_path", "image_paths"])
        true_labels = extract_field_from_prediction(
            prediction, ["gt_label", "gt_labels", "label", "labels"]
        )
        pred_scores = extract_field_from_prediction(
            prediction, ["pred_score", "pred_scores", "anomaly_score"]
        )
        pred_labels = extract_field_from_prediction(prediction, ["pred_label", "pred_labels"])

        paths_array = image_paths
        labels_array = to_numpy_array(true_labels)
        scores_array = to_numpy_array(pred_scores)
        pred_labels_array = to_numpy_array(pred_labels)

        if paths_array is None or labels_array is None or scores_array is None:
            continue

        if not isinstance(paths_array, list):
            try:
                paths_array = list(paths_array)
            except TypeError:
                paths_array = [str(paths_array)]

        for index, image_path in enumerate(paths_array):
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": int(labels_array[index]),
                    "score": float(scores_array[index]),
                    "pred_label": (
                        int(pred_labels_array[index])
                        if pred_labels_array is not None and index < len(pred_labels_array)
                        else None
                    ),
                }
            )

    return rows
