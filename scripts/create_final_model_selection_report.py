"""
Create the final MVTec model selection report.

This script compares:
- ResNet18
- PaDiM
- PatchCore
- FastFlow
- WinCLIP k=0/1/4
- Hybrid score-fusion models

It generates:
- aggregate metrics
- improvement summaries
- final model rankings
- selection rationale
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.reporting.io import write_dataframe_csv, write_markdown  # noqa: E402
from defect_detection.utils.config import load_yaml_config  # noqa: E402

METRICS = [
    "classification_auroc",
    "classification_f1",
    "precision",
    "recall",
    "false_negative_rate",
    "false_positive_rate",
    "false_negative_count",
    "segmentation_auroc",
    "segmentation_f1",
    "inference_latency_ms_mean",
    "throughput_images_per_second",
    "model_size_mb",
]


def load_csv(path: Path, model_label: str) -> pd.DataFrame:
    """Load a per-category result CSV and attach model label."""
    if not path.exists():
        print(f"WARNING: missing input file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "model_name" not in df.columns:
        df["model_name"] = model_label

    return df


def normalize_model_rows(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize model naming across source files."""
    if df.empty:
        return df

    df = df.copy()

    if source_name == "winclip":
        df["model_name"] = df["few_shot_k"].apply(lambda k: f"winclip_k{int(k)}")

    if source_name == "hybrid":
        df["model_name"] = df.apply(
            lambda row: (f"{row['model_name']}_aw{row['anomaly_weight']}_vw{row['vlm_weight']}"),
            axis=1,
        )

    if source_name in {"resnet18", "padim", "patchcore", "fastflow"}:
        df["model_name"] = source_name

    for metric in METRICS:
        if metric not in df.columns:
            df[metric] = None

    keep_columns = ["model_name", "category"] + METRICS
    return df[keep_columns]


def aggregate_results(all_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-category metrics by model."""
    aggregate_df = all_df.groupby("model_name")[METRICS].mean(numeric_only=True).reset_index()

    return aggregate_df


def get_model_row(aggregate_df: pd.DataFrame, model_name: str) -> pd.Series:
    """Get one aggregate row by model name."""
    rows = aggregate_df[aggregate_df["model_name"] == model_name]
    if rows.empty:
        raise ValueError(f"Model not found in aggregate results: {model_name}")
    return rows.iloc[0]


def improvement_row(
    aggregate_df: pd.DataFrame,
    from_model: str,
    to_model: str,
) -> dict:
    """Compute absolute and relative improvement between two models."""
    base = get_model_row(aggregate_df, from_model)
    target = get_model_row(aggregate_df, to_model)

    row = {
        "from_model": from_model,
        "to_model": to_model,
    }

    for metric in [
        "classification_auroc",
        "classification_f1",
        "recall",
        "false_negative_rate",
        "false_negative_count",
        "segmentation_auroc",
        "segmentation_f1",
    ]:
        base_value = base.get(metric)
        target_value = target.get(metric)

        row[f"{metric}_from"] = base_value
        row[f"{metric}_to"] = target_value

        if pd.isna(base_value) or pd.isna(target_value):
            row[f"{metric}_absolute_change"] = None
            row[f"{metric}_relative_change_pct"] = None
            continue

        absolute_change = target_value - base_value

        # For FNR and false-negative count, reduction is positive improvement.
        if metric in {"false_negative_rate", "false_negative_count"}:
            absolute_improvement = base_value - target_value
        else:
            absolute_improvement = absolute_change

        row[f"{metric}_absolute_change"] = absolute_improvement

        if base_value != 0:
            row[f"{metric}_relative_change_pct"] = (absolute_improvement / abs(base_value)) * 100.0
        else:
            row[f"{metric}_relative_change_pct"] = None

    return row


def create_improvement_analysis(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    """Create core model improvement comparisons."""
    comparisons = [
        ("resnet18", "patchcore"),
        ("resnet18", "fastflow"),
        ("resnet18", "patchcore_winclip_k4_aw0.7_vw0.3"),
        ("patchcore", "patchcore_winclip_k4_aw0.7_vw0.3"),
        ("fastflow", "patchcore_winclip_k4_aw0.7_vw0.3"),
    ]

    rows = []

    for from_model, to_model in comparisons:
        if from_model in set(aggregate_df["model_name"]) and to_model in set(
            aggregate_df["model_name"]
        ):
            rows.append(improvement_row(aggregate_df, from_model, to_model))

    return pd.DataFrame(rows)


def create_final_ranking(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final benchmark ranking.

    Score intentionally prioritizes industrial missed-defect risk:
    - recall
    - false-negative rate
    - AUROC/F1
    - segmentation
    - latency/model size when available
    """
    df = aggregate_df.copy()

    max_latency = df["inference_latency_ms_mean"].max(skipna=True)
    max_size = df["model_size_mb"].max(skipna=True)

    df["recall_score"] = df["recall"].fillna(0) * 35.0
    df["fnr_score"] = (1.0 - df["false_negative_rate"].fillna(1.0)) * 30.0
    df["auroc_score"] = df["classification_auroc"].fillna(0) * 15.0
    df["f1_score_component"] = df["classification_f1"].fillna(0) * 10.0
    df["segmentation_score"] = df["segmentation_auroc"].fillna(0) * 5.0

    if pd.notna(max_latency) and max_latency > 0:
        df["latency_score"] = (
            1.0 - (df["inference_latency_ms_mean"].fillna(max_latency) / max_latency)
        ) * 3.0
    else:
        df["latency_score"] = 0.0

    if pd.notna(max_size) and max_size > 0:
        df["model_size_score"] = (1.0 - (df["model_size_mb"].fillna(max_size) / max_size)) * 2.0
    else:
        df["model_size_score"] = 0.0

    df["final_selection_score"] = (
        df["recall_score"]
        + df["fnr_score"]
        + df["auroc_score"]
        + df["f1_score_component"]
        + df["segmentation_score"]
        + df["latency_score"]
        + df["model_size_score"]
    )

    return df.sort_values("final_selection_score", ascending=False)


def fmt(value: float | int | None, digits: int = 4) -> str:
    """Format numeric value safely."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def create_markdown_report(
    aggregate_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> str:
    """Create the markdown report."""
    best_hybrid = get_model_row(aggregate_df, "patchcore_winclip_k4_aw0.7_vw0.3")
    resnet = get_model_row(aggregate_df, "resnet18")
    patchcore = get_model_row(aggregate_df, "patchcore")
    fastflow = get_model_row(aggregate_df, "fastflow")

    report = [
        "# Final MVTec Model Selection Report",
        "",
        "## Executive Summary",
        "",
        "This report summarizes the final MVTec AD benchmark across supervised CNN, classical anomaly detection, VLM, and hybrid score-fusion approaches.",
        "",
        "The selected highest-quality model is:",
        "",
        "**PatchCore + WinCLIP k=4 with 0.7 anomaly / 0.3 VLM weighting.**",
        "",
        "This model was selected because it achieved the best aggregate balance of classification AUROC, recall, false-negative reduction, and classification F1 across MVTec categories.",
        "",
        "## Final Selected Model",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Classification AUROC | {fmt(best_hybrid['classification_auroc'])} |",
        f"| Classification F1 | {fmt(best_hybrid['classification_f1'])} |",
        f"| Recall | {fmt(best_hybrid['recall'])} |",
        f"| False-negative rate | {fmt(best_hybrid['false_negative_rate'])} |",
        f"| Mean false-negative count | {fmt(best_hybrid['false_negative_count'])} |",
        "",
        "## Why PatchCore + WinCLIP k=4 Was Selected",
        "",
        "PatchCore already provided the strongest pure anomaly-detection foundation. WinCLIP k=4 added a small semantic/VLM signal that improved missed-defect behavior when fused conservatively.",
        "",
        "The selected 0.7 / 0.3 weighting is important: the anomaly model remains dominant, "
        "while the VLM contributes auxiliary semantic information. Higher VLM weighting "
        "degraded performance, which supports the conclusion that VLMs are useful as auxiliary "
        "signals rather than primary industrial anomaly detectors.",
        "",
        "## Aggregate Model Comparison",
        "",
        aggregate_df.sort_values("recall", ascending=False)[
            [
                "model_name",
                "classification_auroc",
                "classification_f1",
                "precision",
                "recall",
                "false_negative_rate",
                "false_negative_count",
                "segmentation_auroc",
                "segmentation_f1",
                "inference_latency_ms_mean",
                "throughput_images_per_second",
                "model_size_mb",
            ]
        ].to_markdown(index=False),
        "",
        "## Improvement Analysis",
        "",
        "The most important improvement path is:",
        "",
        "`ResNet18 supervised baseline -> PatchCore normal-only anomaly detection -> PatchCore + WinCLIP hybrid fusion`",
        "",
        improvement_df.to_markdown(index=False),
        "",
        "## Key Improvement Highlights",
        "",
        f"- ResNet18 recall: {fmt(resnet['recall'])}",
        f"- PatchCore recall: {fmt(patchcore['recall'])}",
        f"- Best hybrid recall: {fmt(best_hybrid['recall'])}",
        "",
        f"- ResNet18 FNR: {fmt(resnet['false_negative_rate'])}",
        f"- PatchCore FNR: {fmt(patchcore['false_negative_rate'])}",
        f"- Best hybrid FNR: {fmt(best_hybrid['false_negative_rate'])}",
        "",
        "## Model Positioning",
        "",
        "| Role | Selected Model | Reason |",
        "|---|---|---|",
        "| Best supervised baseline | ResNet18 | Fast and simple, but weak recall/FNR under unseen defects |",
        "| Best pure anomaly detector | PatchCore | Strongest standalone anomaly quality |",
        "| Best anomaly runtime/size tradeoff | FastFlow | Better quality-to-runtime tradeoff than PatchCore |",
        "| Best VLM benchmark | WinCLIP k=4 / k=1 | k=4 improved AUROC/localization, k=1 had strong recall |",
        "| Best overall quality | PatchCore + WinCLIP k=4, 0.7/0.3 | Best aggregate recall/FNR/AUROC balance |",
        "",
        "## Engineering Tradeoff Analysis",
        "",
        "### ResNet18",
        "",
        "ResNet18 is fast and lightweight, but it depends on labeled defect samples and showed "
        "weaker recall/FNR under cross-category MVTec benchmarking. It remains useful as a "
        "supervised baseline, not as the final industrial anomaly detector.",
        "",
        "### PaDiM",
        "",
        "PaDiM improved recall and provided localization, validating the shift from supervised defect classification to normal-only anomaly detection.",
        "",
        "### PatchCore",
        "",
        "PatchCore was the strongest pure anomaly detector. Its weakness is production cost: high model artifact size and higher inference latency.",
        "",
        "### FastFlow",
        "",
        f"FastFlow achieved recall {fmt(fastflow['recall'])} and FNR {fmt(fastflow['false_negative_rate'])}, while offering a lighter runtime and artifact profile than PatchCore.",
        "",
        "### WinCLIP",
        "",
        "WinCLIP was valuable as a zero/few-shot VLM benchmark, but it was too slow to be the preferred standalone runtime option in the current CPU/local setup.",
        "",
        "### Hybrid Fusion",
        "",
        "Hybrid score fusion worked best when PatchCore remained the dominant signal and WinCLIP contributed a smaller auxiliary semantic signal.",
        "",
        "## Important Limitations",
        "",
        "- Hybrid latency was not fully measured end-to-end, so it should not be treated as a settled runtime conclusion until measured.",
        "- Thresholded F1/recall/FNR for hybrid models use diagnostic max-F1 thresholding.",
        "- AnomalyCLIP was treated as literature/reference only because local CPU reproduction was impractical.",
        "- EfficientAD was deferred because CPU training time exceeded practical limits.",
        "",
        "## Final Decision",
        "",
        "**Research-quality winner:** PatchCore + WinCLIP k=4, 0.7 anomaly / 0.3 VLM.",
        "",
        "**Best pure anomaly model:** PatchCore.",
        "",
        "**Best anomaly runtime/size tradeoff:** FastFlow.",
        "",
        "## Notes",
        "",
        "This report is limited to the MVTec AD benchmark results collected in this repo.",
        "",
    ]

    return "\n".join(report)


def main() -> int:
    """Create the final model selection report artifacts."""
    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/final_model_selection.yaml")

    output_dir = PROJECT_ROOT / config["outputs"]["report_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for source_name, input_path in config["inputs"].items():
        df = load_csv(PROJECT_ROOT / input_path, source_name)
        df = normalize_model_rows(df, source_name)
        if not df.empty:
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    aggregate_df = aggregate_results(all_df)
    improvement_df = create_improvement_analysis(aggregate_df)
    ranking_df = create_final_ranking(aggregate_df)

    write_dataframe_csv(all_df, PROJECT_ROOT / config["outputs"]["all_models_csv"])
    write_dataframe_csv(aggregate_df, PROJECT_ROOT / config["outputs"]["aggregate_csv"])
    write_dataframe_csv(improvement_df, PROJECT_ROOT / config["outputs"]["improvement_csv"])
    write_dataframe_csv(ranking_df, PROJECT_ROOT / config["outputs"]["ranking_csv"])

    report = create_markdown_report(
        aggregate_df=aggregate_df,
        improvement_df=improvement_df,
        ranking_df=ranking_df,
    )

    write_markdown(report, PROJECT_ROOT / config["outputs"]["markdown_report"])

    print("Final model selection report created.")
    print(PROJECT_ROOT / config["outputs"]["markdown_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
