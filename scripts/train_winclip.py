"""
Evaluate WinCLIP on one MVTec AD category and one k-shot setting.

Purpose
-------
This script runs Anomalib WinCLIP and exports standardized benchmark results
aligned with metric contract.

Why WinCLIP matters
-------------------
WinCLIP tests zero-shot and few-shot anomaly detection using CLIP embeddings.
Unlike ResNet18 or Anomalib normal-only models, it does not require category-
specific training. It uses language prompts and optional normal reference images.

Run:
    python scripts/train_winclip.py --category bottle --k-shot 0
    python scripts/train_winclip.py --category bottle --k-shot 1
    python scripts/train_winclip.py --category bottle --k-shot 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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
from defect_detection.training.common import set_seed, setup_mlflow  # noqa: E402
from defect_detection.utils.config import load_yaml_config  # noqa: E402
from defect_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def resolve_accelerator(config_value: str) -> str:
    """Resolve Anomalib/Lightning accelerator."""
    if config_value == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"

    return config_value


def normalize_class_name(category: str) -> str:
    """
    Convert MVTec category names into prompt-friendly class names.

    Examples
    --------
    metal_nut -> metal nut
    toothbrush -> toothbrush
    """
    return category.replace("_", " ")


def get_directory_size_mb(directory: Path) -> float:
    """Calculate recursive directory size in MB."""
    if not directory.exists():
        return 0.0

    total_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    return total_bytes / (1024 * 1024)


def benchmark_predict_latency(
    engine: Any,
    model: Any,
    datamodule: Any,
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


def import_winclip_class():
    """
    Import WinClip with version-tolerant paths.

    Anomalib docs show:
        from anomalib.models.image import WinClip

    Some versions may also expose:
        from anomalib.models import WinClip
    """
    try:
        from anomalib.models.image import WinClip

        return WinClip
    except ImportError:
        from anomalib.models import WinClip

        return WinClip


def build_winclip_model(category: str, k_shot: int, config: dict):
    """
    Build WinCLIP model with version-tolerant constructor handling.
    """
    WinClip = import_winclip_class()

    class_name = normalize_class_name(category)
    scales = tuple(config["model"]["scales"])

    constructor_attempts = [
        {
            "class_name": class_name,
            "k_shot": k_shot,
            "scales": scales,
        },
        {
            "class_name": class_name,
            "k_shot": k_shot,
        },
        {
            "k_shot": k_shot,
        },
        {},
    ]

    for kwargs in constructor_attempts:
        try:
            logger.info("Trying WinClip constructor kwargs: %s", kwargs)
            return WinClip(**kwargs)
        except TypeError as error:
            logger.warning("WinClip constructor attempt failed: %s", error)

    raise RuntimeError("Could not construct WinClip model with any known signature.")


def build_engine(config: dict, accelerator: str):
    """
    Build Anomalib Engine with version-compatible arguments.
    """
    from anomalib.engine import Engine

    try:
        return Engine(
            accelerator=accelerator,
            devices=int(config["engine"]["devices"]),
        )
    except TypeError:
        return Engine()


def save_winclip_artifact(
    artifact_dir: Path,
    category: str,
    k_shot: int,
    config: dict,
) -> str:
    """
    Save a lightweight WinCLIP configuration artifact.

    WinCLIP uses pretrained CLIP weights rather than category-trained model weights.
    We save the prompt/few-shot configuration for reproducibility.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = artifact_dir / f"{category}_k{k_shot}_winclip_config.json"

    payload = {
        "model_family": "vlm",
        "model_name": "winclip",
        "category": category,
        "class_name": normalize_class_name(category),
        "k_shot": k_shot,
        "scales": config["model"]["scales"],
        "note": (
            "WinCLIP uses shared pretrained CLIP weights. This artifact records "
            "the category prompt and few-shot configuration, not trained weights."
        ),
    }

    with artifact_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return str(artifact_path.relative_to(PROJECT_ROOT))


def main() -> int:
    """Run WinCLIP for one category and k-shot setting."""
    parser = argparse.ArgumentParser(description="Evaluate WinCLIP benchmark WinCLIP.")
    parser.add_argument("--category", required=True, help="MVTec AD category.")
    parser.add_argument("--k-shot", type=int, required=True, help="0, 1, or 4.")
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/winclip_benchmark.yaml")

    category = args.category
    k_shot = args.k_shot
    seed = int(config["benchmark"]["seed"])

    set_seed(seed)

    dataset_root = PROJECT_ROOT / config["dataset"]["root_dir"]
    raw_metrics_dir = PROJECT_ROOT / config["outputs"]["raw_metrics_dir"] / f"k{k_shot}"
    artifact_dir = PROJECT_ROOT / config["outputs"]["base_artifact_dir"] / category / f"k{k_shot}"

    raw_metrics_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    accelerator = resolve_accelerator(str(config["engine"]["accelerator"]))

    logger.info(
        "Running WinCLIP category=%s k_shot=%d accelerator=%s", category, k_shot, accelerator
    )

    try:
        from anomalib.data import MVTecAD
    except ImportError as error:
        logger.error("Could not import Anomalib MVTecAD: %s", error)
        return 1

    datamodule = MVTecAD(
        root=dataset_root,
        category=category,
        train_batch_size=1,
        eval_batch_size=int(config["engine"]["eval_batch_size"]),
        num_workers=int(config["engine"]["num_workers"]),
    )

    model = build_winclip_model(category=category, k_shot=k_shot, config=config)
    engine = build_engine(config=config, accelerator=accelerator)

    mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
    # mlflow.set_experiment(str(config["mlflow"]["experiment_name"]))
    setup_mlflow(
        str(config["mlflow"]["tracking_uri"]),
        str(config["mlflow"]["experiment_name"]),
    )

    start_time = time.perf_counter()

    with mlflow.start_run(run_name=f"winclip_benchmark_{category}_k{k_shot}"):
        mlflow.log_params(
            {
                "category": category,
                "class_name": normalize_class_name(category),
                "model_family": "vlm",
                "model_name": "winclip",
                "seed": seed,
                "training_mode": "zero_shot" if k_shot == 0 else "few_shot",
                "k_shot": k_shot,
                "scales": ",".join(str(scale) for scale in config["model"]["scales"]),
                "accelerator": accelerator,
                "eval_batch_size": config["engine"]["eval_batch_size"],
            }
        )

        # WinCLIP does not train. Engine.test sets up and evaluates the model.
        test_results = engine.test(model=model, datamodule=datamodule)
        evaluation_time_seconds = time.perf_counter() - start_time

        normalized_metrics = normalize_anomalib_test_results(test_results)

        predictions, latency_mean, latency_p95, throughput = benchmark_predict_latency(
            engine=engine,
            model=model,
            datamodule=datamodule,
        )

        confusion_metrics = compute_confusion_metrics_from_predictions(predictions)

        model_artifact_path = save_winclip_artifact(
            artifact_dir=artifact_dir,
            category=category,
            k_shot=k_shot,
            config=config,
        )

        artifact_size_mb = get_directory_size_mb(artifact_dir)

        classification_f1 = normalized_metrics["classification_f1"]
        if confusion_metrics.get("classification_f1_from_predictions") is not None:
            classification_f1 = float(confusion_metrics["classification_f1_from_predictions"])

        benchmark_result = BenchmarkResult(
            run_id=f"vlm_winclip_{category}_k{k_shot}_seed{seed}",
            model_family="vlm",
            model_name="winclip",
            category=category,
            seed=seed,
            training_mode="zero_shot" if k_shot == 0 else "few_shot",
            few_shot_k=k_shot,
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
            training_time_seconds=0.0,
            inference_latency_ms_mean=latency_mean,
            inference_latency_ms_p95=latency_p95,
            throughput_images_per_second=throughput,
            model_size_mb=artifact_size_mb,
            normal_train_fraction=None,
            defect_train_fraction=0.0,
            model_artifact_path=model_artifact_path,
            report_artifact_path=str(raw_metrics_dir.relative_to(PROJECT_ROOT)),
            notes=(
                "WinCLIP zero/few-shot VLM anomaly detection. Uses shared pretrained "
                "CLIP weights and category-specific prompt/reference configuration."
            ),
        )

        payload = benchmark_result.to_dict()
        payload["evaluation_time_seconds"] = float(evaluation_time_seconds)
        payload["raw_anomalib_test_results"] = test_results

        metrics_path = raw_metrics_dir / f"{category}_winclip_k{k_shot}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, default=str)

        for metric_name, metric_value in benchmark_result.to_dict().items():
            if isinstance(metric_value, int | float):
                mlflow.log_metric(metric_name, float(metric_value))

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(PROJECT_ROOT / model_artifact_path))

        logger.info("Saved metrics: %s", metrics_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
