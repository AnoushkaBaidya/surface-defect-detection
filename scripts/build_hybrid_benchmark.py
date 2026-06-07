"""
Create hybrid score-fusion benchmark.

Fusion rule:
    fused_score = anomaly_weight * normalized_anomaly_score
                + vlm_weight * normalized_vlm_score

The thresholded metrics use a diagnostic max-F1 threshold. AUROC remains the
primary threshold-independent comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = PROJECT_ROOT / "artifacts/benchmarks/hybrid"
RAW_SCORES_DIR = BASE_DIR / "raw_scores"
RAW_METRICS_DIR = BASE_DIR / "raw_metrics"


CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

COMBINATIONS = [
    ("patchcore_winclip_k1", "patchcore", "winclip", 1),
    ("patchcore_winclip_k4", "patchcore", "winclip", 4),
    ("fastflow_winclip_k1", "fastflow", "winclip", 1),
    ("fastflow_winclip_k4", "fastflow", "winclip", 4),
]

WEIGHTS = [
    (0.7, 0.3),
    (0.5, 0.5),
    (0.3, 0.7),
]


def minmax_normalize(values: pd.Series) -> pd.Series:
    """Min-max normalize scores safely."""
    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return pd.Series(np.zeros(len(values)), index=values.index)

    return (values - minimum) / (maximum - minimum)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find threshold maximizing F1."""
    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Compute classification metrics from scores."""
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "classification_auroc": float(roc_auc_score(y_true, scores)),
        "classification_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
    }


def load_scores(model_name: str, category: str, k_shot: int | None = None) -> pd.DataFrame:
    """Load score CSV."""
    if model_name == "winclip":
        path = RAW_SCORES_DIR / model_name / f"k{k_shot}" / f"{category}.csv"
    else:
        path = RAW_SCORES_DIR / model_name / f"{category}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def evaluate_one(
    category: str, combo_name: str, anomaly_model: str, k_shot: int, aw: float, vw: float
) -> dict:
    """Evaluate one hybrid combination."""
    anomaly_df = load_scores(anomaly_model, category)
    vlm_df = load_scores("winclip", category, k_shot)

    merged = anomaly_df.merge(
        vlm_df,
        on=["image_path", "label", "category"],
        suffixes=("_anomaly", "_vlm"),
    )

    if merged.empty:
        raise RuntimeError(f"No matching image paths for {combo_name} {category}")

    merged["anomaly_score_norm"] = minmax_normalize(merged["score_anomaly"])
    merged["vlm_score_norm"] = minmax_normalize(merged["score_vlm"])

    merged["fused_score"] = aw * merged["anomaly_score_norm"] + vw * merged["vlm_score_norm"]

    y_true = merged["label"].astype(int).to_numpy()
    fused_scores = merged["fused_score"].to_numpy()

    threshold = best_f1_threshold(y_true, fused_scores)
    metrics = compute_metrics(y_true, fused_scores, threshold)

    row = {
        "model_family": "hybrid",
        "model_name": combo_name,
        "category": category,
        "anomaly_model": anomaly_model,
        "vlm_model": "winclip",
        "few_shot_k": k_shot,
        "anomaly_weight": aw,
        "vlm_weight": vw,
        "threshold": threshold,
        "inference_latency_ms_mean": None,
        "throughput_images_per_second": None,
        "notes": "Hybrid score fusion using min-max normalized per-image scores.",
    }

    row.update(metrics)

    RAW_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(
        RAW_METRICS_DIR / f"{category}_{combo_name}_aw{aw}_vw{vw}_scores.csv",
        index=False,
    )

    return row


def main() -> int:
    """Run all hybrid fusion evaluations."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for combo_name, anomaly_model, _, k_shot in COMBINATIONS:
        for aw, vw in WEIGHTS:
            for category in CATEGORIES:
                rows.append(
                    evaluate_one(
                        category=category,
                        combo_name=combo_name,
                        anomaly_model=anomaly_model,
                        k_shot=k_shot,
                        aw=aw,
                        vw=vw,
                    )
                )

    df = pd.DataFrame(rows)

    per_category_path = BASE_DIR / "per_category_results.csv"
    aggregate_path = BASE_DIR / "aggregate_results.csv"

    df.to_csv(per_category_path, index=False)

    aggregate_df = (
        df.groupby(["model_name", "anomaly_weight", "vlm_weight"])
        .agg(
            classification_auroc=("classification_auroc", "mean"),
            classification_f1=("classification_f1", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            false_negative_rate=("false_negative_rate", "mean"),
            false_positive_rate=("false_positive_rate", "mean"),
            false_negative_count=("false_negative_count", "mean"),
        )
        .reset_index()
        .sort_values(
            ["recall", "false_negative_rate", "classification_auroc"],
            ascending=[False, True, False],
        )
    )

    aggregate_df.to_csv(aggregate_path, index=False)

    print(f"Saved: {per_category_path}")
    print(f"Saved: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
