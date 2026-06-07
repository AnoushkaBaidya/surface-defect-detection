"""
Train/evaluate PaDiM on one MVTec AD category.

Purpose
-------
This script runs PaDiM using Anomalib and exports standardized benchmark
results aligned with the shared benchmark contract.

Important engineering note
--------------------------
PaDiM is not a conventional epoch-trained classifier. It extracts features from
normal images and estimates patch-level feature distributions. Therefore,
"industrial-grade training" for PaDiM means:
- consistent preprocessing
- controlled category-wise training
- standardized thresholding/evaluation
- latency/throughput measurement
- complete metric/report artifacts

Run:
    python scripts/train_padim.py --category bottle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.evaluation.anomalib_metrics import (  # noqa: E402
    compute_confusion_metrics_from_predictions,
    normalize_anomalib_test_results,
)
from defect_detection.evaluation.benchmark_schema import BenchmarkResult  # noqa: E402
from defect_detection.training.common import (  # noqa: E402
    get_directory_size_mb,
    set_seed,
    setup_mlflow,
)
from defect_detection.utils.config import load_yaml_config  # noqa: E402
from defect_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def resolve_accelerator(config_value: str) -> str:
    """
    Resolve accelerator for Anomalib Engine.

    Parameters
    ----------
    config_value:
        'auto', 'cpu', or 'gpu'.

    Returns
    -------
    str
        Accelerator string accepted by Anomalib/Lightning.
    """
    if config_value == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"

    return config_value


def benchmark_predict_latency(
    engine, model, datamodule
) -> tuple[list[object], float, float, float]:
    """
    Run Anomalib prediction and measure latency/throughput.

    Returns
    -------
    tuple[list[object], float, float, float]
        predictions, mean latency ms/image, p95 latency ms/image, throughput images/sec.
    """
    start = time.perf_counter()
    predictions = engine.predict(model=model, datamodule=datamodule)
    elapsed = time.perf_counter() - start

    # Estimate sample count from prediction batches.
    total_images = 0
    per_batch_latencies = []

    if predictions:
        batch_elapsed = elapsed / len(predictions)

        for prediction in predictions:
            batch_size = 1

            for candidate_field in ["image", "image_path", "gt_label", "gt_labels"]:
                value = getattr(prediction, candidate_field, None)
                if value is None and isinstance(prediction, dict):
                    value = prediction.get(candidate_field)

                if value is not None:
                    try:
                        batch_size = len(value)
                    except TypeError:
                        batch_size = 1
                    break

            total_images += batch_size
            per_image_ms = (batch_elapsed / max(batch_size, 1)) * 1000.0
            per_batch_latencies.extend([per_image_ms] * batch_size)

    if total_images == 0:
        return predictions or [], 0.0, 0.0, 0.0

    mean_latency = float(np.mean(per_batch_latencies))
    p95_latency = float(np.percentile(per_batch_latencies, 95))
    throughput = float(total_images / elapsed) if elapsed > 0 else 0.0

    return predictions or [], mean_latency, p95_latency, throughput


def save_model_artifact(engine, model, model_dir: Path) -> str | None:
    """
    Save Anomalib model checkpoint if supported by current Engine version.

    Parameters
    ----------
    engine:
        Anomalib Engine.
    model:
        Anomalib model.
    model_dir:
        Output model directory.

    Returns
    -------
    str | None
        Relative checkpoint path if saved.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "padim_model.ckpt"

    try:
        if hasattr(engine, "trainer") and hasattr(engine.trainer, "save_checkpoint"):
            engine.trainer.save_checkpoint(str(checkpoint_path))
            return str(checkpoint_path.relative_to(PROJECT_ROOT))
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not save Anomalib checkpoint via trainer: %s", error)

    try:
        torch.save(model.state_dict(), model_dir / "padim_state_dict.pt")
        return str((model_dir / "padim_state_dict.pt").relative_to(PROJECT_ROOT))
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not save model state_dict: %s", error)

    return None


def main() -> int:
    """Run PaDiM for one category."""
    parser = argparse.ArgumentParser(description="Train/evaluate PaDiM benchmark PaDiM.")
    parser.add_argument("--category", required=True, help="MVTec AD category.")
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/padim_benchmark.yaml")

    category = args.category
    seed = int(config["benchmark"]["seed"])
    set_seed(seed)

    dataset_root = PROJECT_ROOT / config["dataset"]["root_dir"]
    model_dir = PROJECT_ROOT / config["outputs"]["base_model_dir"] / category
    raw_metrics_dir = PROJECT_ROOT / config["outputs"]["raw_metrics_dir"]
    raw_metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    accelerator = resolve_accelerator(str(config["engine"]["accelerator"]))

    logger.info("Running PaDiM category=%s accelerator=%s", category, accelerator)

    try:
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Padim
    except ImportError as error:
        logger.error("Could not import Anomalib components: %s", error)
        logger.error("Run: pip install anomalib")
        return 1

    datamodule = MVTecAD(
        root=dataset_root,
        category=category,
        train_batch_size=int(config["engine"]["train_batch_size"]),
        eval_batch_size=int(config["engine"]["eval_batch_size"]),
        num_workers=int(config["engine"]["num_workers"]),
    )

    # Keep constructor conservative for version compatibility.
    # Some Anomalib versions expose backbone/layers directly; if unsupported,
    # default PaDiM settings still run.
    try:
        model = Padim(
            backbone=str(config["model"]["backbone"]),
            layers=tuple(config["model"]["layers"]),
        )
    except TypeError:
        logger.warning("Installed Anomalib Padim constructor did not accept backbone/layers.")
        logger.warning("Falling back to default Padim().")
        model = Padim()

    engine = Engine(
        accelerator=accelerator,
        devices=int(config["engine"]["devices"]),
        max_epochs=int(config["engine"]["max_epochs"]),
        default_root_dir=str(model_dir),
    )

    mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
    # mlflow.set_experiment(str(config["mlflow"]["experiment_name"]))
    setup_mlflow(
        str(config["mlflow"]["tracking_uri"]),
        str(config["mlflow"]["experiment_name"]),
    )

    fit_start = time.perf_counter()

    with mlflow.start_run(run_name=f"padim_benchmark_{category}"):
        mlflow.log_params(
            {
                "category": category,
                "model_family": "anomalib",
                "model_name": "padim",
                "seed": seed,
                "training_mode": "normal_only",
                "backbone": config["model"]["backbone"],
                "layers": ",".join(config["model"]["layers"]),
                "accelerator": accelerator,
                "train_batch_size": config["engine"]["train_batch_size"],
                "eval_batch_size": config["engine"]["eval_batch_size"],
                "max_epochs": config["engine"]["max_epochs"],
            }
        )

        engine.fit(model=model, datamodule=datamodule)
        training_time_seconds = time.perf_counter() - fit_start

        test_results = engine.test(model=model, datamodule=datamodule)
        normalized_metrics = normalize_anomalib_test_results(test_results)

        predictions, latency_mean, latency_p95, throughput = benchmark_predict_latency(
            engine=engine,
            model=model,
            datamodule=datamodule,
        )

        confusion_metrics = compute_confusion_metrics_from_predictions(predictions)

        model_artifact_path = save_model_artifact(
            engine=engine,
            model=model,
            model_dir=model_dir,
        )

        model_size_mb = get_directory_size_mb(model_dir)

        classification_f1 = normalized_metrics["classification_f1"]
        if confusion_metrics.get("classification_f1_from_predictions") is not None:
            classification_f1 = float(confusion_metrics["classification_f1_from_predictions"])

        benchmark_result = BenchmarkResult(
            run_id=f"anomalib_padim_{category}_seed{seed}",
            model_family="anomalib",
            model_name="padim",
            category=category,
            seed=seed,
            training_mode="normal_only",
            few_shot_k=None,
            classification_auroc=normalized_metrics["classification_auroc"],
            classification_f1=classification_f1,
            precision=confusion_metrics["precision"],
            recall=confusion_metrics["recall"],
            false_positive_rate=confusion_metrics["false_positive_rate"],
            false_negative_rate=confusion_metrics["false_negative_rate"],
            false_negative_count=confusion_metrics["false_negative_count"],
            false_positive_count=confusion_metrics["false_positive_count"],
            segmentation_auroc=normalized_metrics["segmentation_auroc"],
            segmentation_f1=normalized_metrics["segmentation_f1"],
            segmentation_aupro=normalized_metrics["segmentation_aupro"],
            training_time_seconds=float(training_time_seconds),
            inference_latency_ms_mean=latency_mean,
            inference_latency_ms_p95=latency_p95,
            throughput_images_per_second=throughput,
            model_size_mb=model_size_mb,
            normal_train_fraction=1.0,
            defect_train_fraction=0.0,
            model_artifact_path=model_artifact_path,
            report_artifact_path=str(raw_metrics_dir.relative_to(PROJECT_ROOT)),
            notes=(
                "PaDiM normal-only anomaly detection. Recall/FNR are extracted from "
                "Anomalib predictions when prediction labels are available."
            ),
        )

        payload = benchmark_result.to_dict()
        payload["raw_anomalib_test_results"] = test_results

        metrics_path = raw_metrics_dir / f"{category}_padim_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, default=str)

        for metric_name, metric_value in benchmark_result.to_dict().items():
            if isinstance(metric_value, int | float):
                mlflow.log_metric(metric_name, float(metric_value))

        mlflow.log_artifact(str(metrics_path))
        if model_artifact_path is not None:
            mlflow.log_artifact(str(PROJECT_ROOT / model_artifact_path))

        logger.info("Saved metrics: %s", metrics_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
