"""
Train/evaluate FastFlow on one MVTec AD category.

Purpose
-------
This script runs FastFlow using Anomalib and exports standardized benchmark
results aligned with the shared benchmark contract.

Engineering note
----------------
FastFlow is a trainable normalizing-flow anomaly detection model. Unlike PaDiM
and PatchCore, it uses iterative optimization. This benchmark uses:

- max_epochs
- early stopping
- checkpointing
- standardized train/test/predict metrics
- latency and throughput measurement

Run:
    python scripts/train_fastflow.py --category bottle
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
    """Resolve Anomalib/Lightning accelerator."""
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

    total_images = 0
    per_image_latencies_ms: list[float] = []

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
            per_image_latencies_ms.extend([per_image_ms] * batch_size)

    if total_images == 0:
        return predictions or [], 0.0, 0.0, 0.0

    mean_latency = float(np.mean(per_image_latencies_ms))
    p95_latency = float(np.percentile(per_image_latencies_ms, 95))
    throughput = float(total_images / elapsed) if elapsed > 0 else 0.0

    return predictions or [], mean_latency, p95_latency, throughput


def save_model_artifact(engine, model, model_dir: Path) -> str | None:
    """
    Save FastFlow checkpoint or state dict.

    Returns
    -------
    str | None
        Project-relative artifact path if saved.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "fastflow_model.ckpt"

    try:
        if hasattr(engine, "trainer") and hasattr(engine.trainer, "save_checkpoint"):
            engine.trainer.save_checkpoint(str(checkpoint_path))
            return str(checkpoint_path.relative_to(PROJECT_ROOT))
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not save checkpoint via trainer: %s", error)

    try:
        torch.save(model.state_dict(), model_dir / "fastflow_state_dict.pt")
        return str((model_dir / "fastflow_state_dict.pt").relative_to(PROJECT_ROOT))
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not save FastFlow state_dict: %s", error)

    return None


def build_fastflow_model(config: dict):
    """
    Build FastFlow model with version-tolerant constructor handling.

    Anomalib constructor signatures can vary. We try the configured
    arguments first, then fall back to defaults if necessary.
    """
    from anomalib.models import Fastflow

    try:
        return Fastflow(
            backbone=str(config["model"]["backbone"]),
            pre_trained=bool(config["model"]["pre_trained"]),
            flow_steps=int(config["model"]["flow_steps"]),
            conv3x3_only=bool(config["model"]["conv3x3_only"]),
            hidden_ratio=float(config["model"]["hidden_ratio"]),
        )
    except TypeError:
        logger.warning("Installed Fastflow constructor did not accept full config.")
        logger.warning("Falling back to default Fastflow().")
        return Fastflow()


def build_callbacks(config: dict, model_dir: Path) -> list[object]:
    """
    Build Lightning callbacks for early stopping and checkpointing.

    Returns an empty list if callback imports or callback construction fail.
    """
    callbacks: list[object] = []

    try:
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        try:
            from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        except ImportError:
            logger.warning("Could not import Lightning callbacks. Continuing without callbacks.")
            return callbacks

    if bool(config["early_stopping"]["enabled"]):
        callbacks.append(
            EarlyStopping(
                monitor=str(config["early_stopping"]["monitor"]),
                mode=str(config["early_stopping"]["mode"]),
                patience=int(config["early_stopping"]["patience"]),
                min_delta=float(config["early_stopping"]["min_delta"]),
            )
        )

    if bool(config["checkpointing"]["enabled"]):
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(model_dir),
                filename="best_fastflow",
                monitor=str(config["checkpointing"]["monitor"]),
                mode=str(config["checkpointing"]["mode"]),
                save_top_k=1,
            )
        )

    return callbacks


def build_engine(config: dict, accelerator: str, model_dir: Path):
    """
    Build Anomalib Engine with callback compatibility fallback.
    """
    from anomalib.engine import Engine

    callbacks = build_callbacks(config=config, model_dir=model_dir)

    try:
        return Engine(
            accelerator=accelerator,
            devices=int(config["engine"]["devices"]),
            max_epochs=int(config["engine"]["max_epochs"]),
            callbacks=callbacks,
            default_root_dir=str(model_dir),
        )
    except TypeError:
        logger.warning("Installed Engine did not accept callbacks/default_root_dir.")
        logger.warning("Falling back to Engine without callbacks.")
        return Engine(
            accelerator=accelerator,
            devices=int(config["engine"]["devices"]),
            max_epochs=int(config["engine"]["max_epochs"]),
        )


def main() -> int:
    """Run FastFlow for one category."""
    parser = argparse.ArgumentParser(description="Train/evaluate FastFlow benchmark FastFlow.")
    parser.add_argument("--category", required=True, help="MVTec AD category.")
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/fastflow_benchmark.yaml")

    category = args.category
    seed = int(config["benchmark"]["seed"])
    set_seed(seed)

    dataset_root = PROJECT_ROOT / config["dataset"]["root_dir"]
    model_dir = PROJECT_ROOT / config["outputs"]["base_model_dir"] / category
    raw_metrics_dir = PROJECT_ROOT / config["outputs"]["raw_metrics_dir"]

    model_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics_dir.mkdir(parents=True, exist_ok=True)

    accelerator = resolve_accelerator(str(config["engine"]["accelerator"]))

    logger.info("Running FastFlow category=%s accelerator=%s", category, accelerator)

    try:
        from anomalib.data import MVTecAD
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

    model = build_fastflow_model(config)
    engine = build_engine(config=config, accelerator=accelerator, model_dir=model_dir)

    mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
    # mlflow.set_experiment(str(config["mlflow"]["experiment_name"]))
    setup_mlflow(
        str(config["mlflow"]["tracking_uri"]),
        str(config["mlflow"]["experiment_name"]),
    )

    fit_start = time.perf_counter()

    with mlflow.start_run(run_name=f"fastflow_benchmark_{category}"):
        mlflow.log_params(
            {
                "category": category,
                "model_family": "anomalib",
                "model_name": "fastflow",
                "seed": seed,
                "training_mode": "normal_only_trainable_density_model",
                "backbone": config["model"]["backbone"],
                "flow_steps": config["model"]["flow_steps"],
                "hidden_ratio": config["model"]["hidden_ratio"],
                "accelerator": accelerator,
                "train_batch_size": config["engine"]["train_batch_size"],
                "eval_batch_size": config["engine"]["eval_batch_size"],
                "max_epochs": config["engine"]["max_epochs"],
                "early_stopping_enabled": config["early_stopping"]["enabled"],
                "early_stopping_monitor": config["early_stopping"]["monitor"],
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
            run_id=f"anomalib_fastflow_{category}_seed{seed}",
            model_family="anomalib",
            model_name="fastflow",
            category=category,
            seed=seed,
            training_mode="normal_only_trainable_density_model",
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
                "FastFlow trainable normalizing-flow anomaly detection. "
                "Early stopping/checkpointing used when supported by installed Anomalib/Lightning version."
            ),
        )

        payload = benchmark_result.to_dict()
        payload["raw_anomalib_test_results"] = test_results

        metrics_path = raw_metrics_dir / f"{category}_fastflow_metrics.json"
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
