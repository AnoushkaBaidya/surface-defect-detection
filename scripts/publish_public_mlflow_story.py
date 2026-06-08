"""
Sanitize and publish the public MLflow history for the MVTec benchmark story.

This utility keeps the checked-in ``mlruns/`` directory aligned with the
current repository narrative:

- MVTec AD only
- ResNet-18 baseline
- PaDiM / PatchCore / FastFlow anomaly baselines
- WinCLIP VLM benchmark
- final model-selection comparison
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import mlflow
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
REPORTS_DIR = PROJECT_ROOT / "reports"

UNWANTED_EXPERIMENT_DIRS = [
    ".trash",
    "202493981775140912",
    "217137935316631074",
    "462589834481428410",
    "550158285621900407",
]

EXPERIMENT_SPECS = {
    "526830661020181409": {
        "experiment_name": "resnet18_mvtec_benchmark",
        "run_prefix_old": "phase3_0_resnet18_",
        "run_prefix_new": "resnet18_benchmark_",
        "source_old": "scripts/train_phase3_0_resnet18.py",
        "source_new": "scripts/train_resnet18.py",
        "text_replacements": {
            "artifacts\\models\\phase3\\resnet18\\": "artifacts\\models\\benchmarks\\resnet18\\",
            "artifacts\\reports\\phase3\\resnet18\\raw_metrics": "artifacts\\benchmarks\\resnet18\\raw_metrics",
        },
    },
    "142296954129725644": {
        "experiment_name": "padim_mvtec_benchmark",
        "run_prefix_old": "phase3_1_padim_",
        "run_prefix_new": "padim_benchmark_",
        "source_old": "scripts/train_phase3_1_padim.py",
        "source_new": "scripts/train_padim.py",
        "text_replacements": {
            "artifacts\\models\\phase3\\padim\\": "artifacts\\models\\benchmarks\\padim\\",
            "artifacts\\reports\\phase3\\padim\\raw_metrics": "artifacts\\benchmarks\\padim\\raw_metrics",
        },
    },
    "844624909902585835": {
        "experiment_name": "patchcore_mvtec_benchmark",
        "run_prefix_old": "phase3_2_patchcore_",
        "run_prefix_new": "patchcore_benchmark_",
        "source_old": "scripts/train_phase3_2_patchcore.py",
        "source_new": "scripts/train_patchcore.py",
        "text_replacements": {
            "artifacts\\models\\phase3\\patchcore\\": "artifacts\\models\\benchmarks\\patchcore\\",
            "artifacts\\reports\\phase3\\patchcore\\raw_metrics": "artifacts\\benchmarks\\patchcore\\raw_metrics",
        },
    },
    "154491621225953668": {
        "experiment_name": "fastflow_mvtec_benchmark",
        "run_prefix_old": "phase3_3_fastflow_",
        "run_prefix_new": "fastflow_benchmark_",
        "source_old": "scripts/train_phase3_3_fastflow.py",
        "source_new": "scripts/train_fastflow.py",
        "text_replacements": {
            "artifacts\\models\\phase3\\fastflow\\": "artifacts\\models\\benchmarks\\fastflow\\",
            "artifacts\\reports\\phase3\\fastflow\\raw_metrics": "artifacts\\benchmarks\\fastflow\\raw_metrics",
        },
    },
    "676412444936204945": {
        "experiment_name": "winclip_mvtec_benchmark",
        "run_prefix_old": "phase3_6_winclip_",
        "run_prefix_new": "winclip_benchmark_",
        "source_old": "scripts/train_phase3_6_winclip.py",
        "source_new": "scripts/train_winclip.py",
        "text_replacements": {
            "artifacts\\models\\phase3\\winclip\\": "artifacts\\models\\benchmarks\\winclip\\",
            "artifacts\\reports\\phase3\\winclip\\raw_metrics": "artifacts\\benchmarks\\winclip\\raw_metrics",
        },
    },
}

PUBLIC_MODEL_SUMMARY = [
    {
        "run_name": "resnet18_baseline",
        "model_name": "resnet18",
        "model_family": "cnn",
        "selection_role": "supervised_baseline",
        "classification_auroc": 0.8574,
        "classification_f1": 0.7499,
        "precision": 0.9471,
        "recall": 0.6905,
        "false_negative_rate": 0.3095,
        "throughput_images_per_second": 20.88,
        "inference_latency_ms_mean": 50.8,
        "model_size_mb": 42.7,
        "source_report": "reports/resnet18_baseline_report.md",
    },
    {
        "run_name": "patchcore_quality_anchor",
        "model_name": "patchcore",
        "model_family": "anomaly_detection",
        "selection_role": "best_pure_anomaly_model",
        "classification_auroc": 0.9823,
        "classification_f1": 0.9710,
        "recall": 0.9723,
        "false_negative_rate": 0.0277,
        "segmentation_auroc": 0.9801,
        "segmentation_f1": 0.5817,
        "inference_latency_ms_mean": 1193.8,
        "model_size_mb": 781.2,
        "source_report": "reports/patchcore_benchmark_report.md",
    },
    {
        "run_name": "fastflow_runtime_tradeoff",
        "model_name": "fastflow",
        "model_family": "anomaly_detection",
        "selection_role": "production_feasible_model",
        "classification_auroc": 0.9336,
        "classification_f1": 0.9449,
        "recall": 0.9606,
        "false_negative_rate": 0.0394,
        "segmentation_auroc": 0.9715,
        "segmentation_f1": 0.5387,
        "throughput_images_per_second": 2.05,
        "inference_latency_ms_mean": 502.7,
        "model_size_mb": 268.7,
        "source_report": "reports/fastflow_benchmark_report.md",
    },
    {
        "run_name": "winclip_k1_recall_variant",
        "model_name": "winclip_k1",
        "model_family": "vlm",
        "selection_role": "best_vlm_recall_variant",
        "classification_auroc": 0.8976,
        "classification_f1": 0.9239,
        "recall": 0.9597,
        "false_negative_rate": 0.0403,
        "segmentation_auroc": 0.9057,
        "segmentation_f1": 0.3768,
        "inference_latency_ms_mean": 19553.3,
        "source_report": "reports/winclip_evaluation_report.md",
    },
    {
        "run_name": "winclip_k4_quality_variant",
        "model_name": "winclip_k4",
        "model_family": "vlm",
        "selection_role": "best_vlm_quality_variant",
        "classification_auroc": 0.9249,
        "classification_f1": 0.9258,
        "recall": 0.9342,
        "false_negative_rate": 0.0658,
        "segmentation_auroc": 0.9121,
        "segmentation_f1": 0.3983,
        "inference_latency_ms_mean": 17231.8,
        "source_report": "reports/winclip_evaluation_report.md",
    },
    {
        "run_name": "patchcore_winclip_k4_hybrid",
        "model_name": "patchcore_winclip_k4_aw0.7_vw0.3",
        "model_family": "hybrid",
        "selection_role": "best_overall_model",
        "classification_auroc": 0.9853,
        "classification_f1": 0.9773,
        "precision": 0.9731,
        "recall": 0.9823,
        "false_negative_rate": 0.0177,
        "false_negative_count": 1.53,
        "source_report": "reports/hybrid_model_report.md",
    },
]


def replace_prefixed_line(text: str, key: str, value: str) -> str:
    """Replace a simple ``key: value`` line inside MLflow meta files."""
    lines = []
    for line in text.splitlines():
        if line.startswith(f"{key}:"):
            lines.append(f"{key}: {value}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def sanitize_text_file(path: Path, replacements: dict[str, str]) -> None:
    """Apply literal text replacements if the file exists."""
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)

    if updated != text:
        path.write_text(updated, encoding="utf-8")


def replace_strings_in_object(value: object, replacements: dict[str, str]) -> object:
    """Recursively replace substrings inside JSON-compatible values."""
    if isinstance(value, str):
        updated = value
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        return updated

    if isinstance(value, list):
        return [replace_strings_in_object(item, replacements) for item in value]

    if isinstance(value, dict):
        return {
            key: replace_strings_in_object(item, replacements)
            for key, item in value.items()
        }

    return value


def sanitize_json_file(path: Path, replacements: dict[str, str]) -> None:
    """Apply replacements inside JSON values while preserving valid JSON."""
    if not path.exists():
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    updated = replace_strings_in_object(payload, replacements)
    path.write_text(json.dumps(updated, indent=2), encoding="utf-8")


def sanitize_kept_experiments() -> None:
    """Rename public experiments and runs and rewrite phase-era paths."""
    for experiment_id, spec in EXPERIMENT_SPECS.items():
        experiment_dir = MLRUNS_DIR / experiment_id
        if not experiment_dir.exists():
            continue

        experiment_meta = experiment_dir / "meta.yaml"
        experiment_meta_text = experiment_meta.read_text(encoding="utf-8")
        experiment_meta_text = replace_prefixed_line(
            experiment_meta_text,
            "name",
            spec["experiment_name"],
        )
        experiment_meta.write_text(experiment_meta_text, encoding="utf-8")

        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name in {"datasets"}:
                continue

            run_meta = run_dir / "meta.yaml"
            if run_meta.exists():
                run_meta_text = run_meta.read_text(encoding="utf-8")
                for line in run_meta_text.splitlines():
                    if line.startswith("run_name: "):
                        old_run_name = line.split("run_name: ", maxsplit=1)[1]
                        new_run_name = old_run_name.replace(
                            spec["run_prefix_old"],
                            spec["run_prefix_new"],
                        )
                        run_meta_text = replace_prefixed_line(run_meta_text, "run_name", new_run_name)
                        break
                run_meta.write_text(run_meta_text, encoding="utf-8")

            tag_replacements = {
                spec["source_old"]: spec["source_new"],
                spec["run_prefix_old"]: spec["run_prefix_new"],
                **spec["text_replacements"],
            }

            sanitize_text_file(run_dir / "tags" / "mlflow.source.name", tag_replacements)
            sanitize_text_file(run_dir / "tags" / "mlflow.runName", tag_replacements)

            artifacts_dir = run_dir / "artifacts"
            if artifacts_dir.exists():
                for json_path in artifacts_dir.rglob("*.json"):
                    sanitize_json_file(json_path, spec["text_replacements"])


def prune_out_of_scope_experiments() -> None:
    """Remove MLflow runs that do not belong in the public MVTec story."""
    for relative_name in UNWANTED_EXPERIMENT_DIRS:
        target = MLRUNS_DIR / relative_name
        if target.exists():
            shutil.rmtree(target)

    for meta_path in MLRUNS_DIR.glob("*/meta.yaml"):
        text = meta_path.read_text(encoding="utf-8")
        if "name: final_mvtec_model_selection" in text:
            shutil.rmtree(meta_path.parent)


def build_public_summary_dataframe() -> pd.DataFrame:
    """Return the public model-comparison dataframe used for summary runs."""
    dataframe = pd.DataFrame(PUBLIC_MODEL_SUMMARY)
    output_path = REPORTS_DIR / "mlflow_model_comparison.csv"
    dataframe.to_csv(output_path, index=False)
    return dataframe


def render_comparison_chart(dataframe: pd.DataFrame) -> Path:
    """Render a compact comparison chart for README/docs references."""
    output_path = REPORTS_DIR / "mlflow_experiment_comparison.png"

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = dataframe["model_name"].tolist()

    axes[0].bar(labels, dataframe["recall"], color="#3b82f6")
    axes[0].set_title("Recall")
    axes[0].set_ylim(0.65, 1.0)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(labels, dataframe["false_negative_rate"], color="#ef4444")
    axes[1].set_title("False-negative rate")
    axes[1].set_ylim(0.0, 0.35)
    axes[1].tick_params(axis="x", rotation=45)

    latency_series = dataframe["inference_latency_ms_mean"].fillna(0.0)
    axes[2].bar(labels, latency_series, color="#10b981")
    axes[2].set_title("Latency (ms)")
    axes[2].tick_params(axis="x", rotation=45)

    figure.suptitle("MLflow Public Comparison Snapshot")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return output_path


def publish_model_selection_experiment(dataframe: pd.DataFrame, chart_path: Path) -> None:
    """Create clean public MLflow runs for final model selection comparison."""
    tracking_uri = MLRUNS_DIR.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("final_mvtec_model_selection")

    for row in dataframe.to_dict(orient="records"):
        run_name = row["run_name"]
        source_report = row.pop("source_report")

        params = {
            "dataset": "mvtec_ad",
            "comparison_scope": "published_repo_summary",
            "source_report": source_report,
            "selection_role": row["selection_role"],
            "model_family": row["model_family"],
            "model_name": row["model_name"],
        }
        metrics = {
            key: float(value)
            for key, value in row.items()
            if key
            not in {"run_name", "model_name", "model_family", "selection_role"}
            and value is not None
        }

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

    best_model_name = "patchcore_winclip_k4_aw0.7_vw0.3"
    best_row = dataframe.loc[dataframe["model_name"] == best_model_name].iloc[0].to_dict()

    with mlflow.start_run(run_name="model_selection_summary"):
        mlflow.log_params(
            {
                "dataset": "mvtec_ad",
                "selected_model": best_model_name,
                "selection_basis": "recall_fnr_auroc_f1_tradeoff",
                "source_report": "reports/final_model_selection_report.md",
            }
        )
        mlflow.log_metrics(
            {
                "selected_model_classification_auroc": float(best_row["classification_auroc"]),
                "selected_model_classification_f1": float(best_row["classification_f1"]),
                "selected_model_recall": float(best_row["recall"]),
                "selected_model_false_negative_rate": float(best_row["false_negative_rate"]),
            }
        )
        mlflow.log_artifact(str(REPORTS_DIR / "mlflow_model_comparison.csv"))
        mlflow.log_artifact(str(chart_path))
        mlflow.log_artifact(str(REPORTS_DIR / "final_model_selection_report.md"))
        mlflow.log_artifact(str(REPORTS_DIR / "hybrid_model_report.md"))


def main() -> int:
    """Clean public MLflow history and publish the final comparison story."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prune_out_of_scope_experiments()
    sanitize_kept_experiments()
    dataframe = build_public_summary_dataframe()
    chart_path = render_comparison_chart(dataframe)
    publish_model_selection_experiment(dataframe, chart_path)
    trash_dir = MLRUNS_DIR / ".trash"
    if trash_dir.exists():
        shutil.rmtree(trash_dir)
    print("Public MLflow history updated.")
    print(chart_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
