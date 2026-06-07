"""Create benchmark visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks" / "summary"

PLOTS_DIR = REPORT_DIR / "plots"


def create_bar_plot(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_name: str,
) -> None:
    """
    Create standardized bar plot.
    """
    plt.figure(figsize=(10, 6))

    plt.bar(
        df["model_name"],
        df[metric],
    )

    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.title(f"{ylabel} Comparison")

    plt.tight_layout()

    output_path = PLOTS_DIR / output_name

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_latency_vs_auroc_plot(
    df: pd.DataFrame,
) -> None:
    """
    Create latency vs AUROC tradeoff plot.
    """
    plt.figure(figsize=(10, 7))

    plt.scatter(
        df["inference_latency_ms_mean"],
        df["classification_auroc"],
        s=250,
    )

    for _, row in df.iterrows():
        plt.text(
            row["inference_latency_ms_mean"],
            row["classification_auroc"],
            row["model_name"],
        )

    plt.xlabel("Inference Latency (ms)")
    plt.ylabel("Classification AUROC")
    plt.title("Latency vs Classification AUROC")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "latency_vs_auroc.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """
    Generate benchmark visualizations.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    aggregate_df = pd.read_csv(REPORT_DIR / "aggregate_model_comparison.csv")

    create_bar_plot(
        aggregate_df,
        metric="classification_auroc",
        ylabel="Classification AUROC",
        output_name="classification_auroc_comparison.png",
    )

    create_bar_plot(
        aggregate_df,
        metric="recall",
        ylabel="Recall",
        output_name="recall_comparison.png",
    )

    create_bar_plot(
        aggregate_df,
        metric="false_negative_rate",
        ylabel="False Negative Rate",
        output_name="fnr_comparison.png",
    )

    create_bar_plot(
        aggregate_df,
        metric="segmentation_auroc",
        ylabel="Segmentation AUROC",
        output_name="segmentation_auroc_comparison.png",
    )

    create_bar_plot(
        aggregate_df,
        metric="inference_latency_ms_mean",
        ylabel="Inference Latency (ms)",
        output_name="latency_comparison.png",
    )

    create_bar_plot(
        aggregate_df,
        metric="model_size_mb",
        ylabel="Model Size (MB)",
        output_name="model_size_comparison.png",
    )

    create_latency_vs_auroc_plot(
        aggregate_df,
    )

    print("\nVisualization generation complete.")
    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
