"""
Classification metric utilities for the CNN baseline.

Why this file exists
--------------------
claims such as recall improvement and false-negative reduction must be
supported by reproducible metrics. This file centralizes metric calculation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
) -> dict[str, float | int]:
    """
    Compute binary classification metrics.

    Positive class:
    - defective = 1

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted class labels.
    y_score:
        Predicted probability/score for positive class.

    Returns
    -------
    dict[str, float | int]
        Metrics dictionary.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_score_array = np.asarray(y_score)

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true_array, y_score_array))
    except ValueError:
        metrics["auroc"] = 0.0

    return metrics
