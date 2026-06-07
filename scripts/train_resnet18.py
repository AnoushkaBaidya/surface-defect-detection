"""
Train one category-specific ResNet18 model.

Purpose
-------
This script trains a supervised CNN baseline for one MVTec category using the
standard ResNet18 benchmark split files.

Industrial training choices
---------------------------
- pretrained ResNet18 backbone
- class-weighted loss for imbalance
- AdamW optimizer
- ReduceLROnPlateau scheduler
- early stopping
- validation recall as primary monitor
- F1 as tie-breaker
- all results logged to MLflow
- standardized metric schema aligned with the shared benchmark contract

Run:
    python scripts/train_resnet18.py --category bottle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.data.classification_dataset import MVTecClassificationDataset  # noqa: E402
from defect_detection.evaluation.benchmark_schema import BenchmarkResult  # noqa: E402
from defect_detection.evaluation.classification_metrics import (  # noqa: E402
    compute_binary_classification_metrics,
)
from defect_detection.training.cnn_model import build_resnet18_binary_classifier  # noqa: E402
from defect_detection.training.common import set_seed, setup_mlflow  # noqa: E402
from defect_detection.utils.config import load_yaml_config  # noqa: E402
from defect_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def get_device(device_config: str) -> torch.device:
    """Resolve device from config."""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_config)


def build_dataloader(
    dataframe: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
    is_train: bool,
) -> DataLoader:
    """Build a PyTorch DataLoader from a split dataframe."""
    dataset = MVTecClassificationDataset(
        dataframe=dataframe,
        project_root=PROJECT_ROOT,
        image_size=image_size,
        is_train=is_train,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
    )


def compute_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Compute class weights for CrossEntropyLoss.

    This reduces bias toward the majority class.
    """
    label_counts = train_df["label"].value_counts().to_dict()

    normal_count = max(int(label_counts.get("normal", 0)), 1)
    defective_count = max(int(label_counts.get("defective", 0)), 1)
    total_count = normal_count + defective_count

    return torch.tensor(
        [
            total_count / (2 * normal_count),
            total_count / (2 * defective_count),
        ],
        dtype=torch.float32,
        device=device,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float | int], list[int], list[int], list[float]]:
    """
    Evaluate model and return loss, metrics, labels, predictions, and scores.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_scores: list[float] = []

    for images, labels in tqdm(dataloader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probabilities = torch.softmax(logits, dim=1)
        defect_scores = probabilities[:, 1]
        predictions = torch.argmax(probabilities, dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_scores.extend(defect_scores.cpu().tolist())

    average_loss = total_loss / max(total_samples, 1)
    metrics = compute_binary_classification_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
        y_score=all_scores,
    )

    return average_loss, metrics, all_labels, all_predictions, all_scores


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    warmup_batches: int = 2,
) -> tuple[float, float, float]:
    """
    Benchmark inference latency and throughput.

    Returns
    -------
    tuple[float, float, float]
        mean latency ms/image, p95 latency ms/image, throughput images/sec.
    """
    model.eval()

    per_image_latencies_ms: list[float] = []
    total_images = 0
    total_elapsed = 0.0

    for batch_index, (images, _) in enumerate(dataloader):
        images = images.to(device)

        if batch_index < warmup_batches:
            _ = model(images)
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        batch_size = images.size(0)

        per_image_latency_ms = (elapsed / batch_size) * 1000.0
        per_image_latencies_ms.extend([per_image_latency_ms] * batch_size)

        total_elapsed += elapsed
        total_images += batch_size

    if not per_image_latencies_ms or total_elapsed == 0:
        return 0.0, 0.0, 0.0

    mean_latency = float(np.mean(per_image_latencies_ms))
    p95_latency = float(np.percentile(per_image_latencies_ms, 95))
    throughput = float(total_images / total_elapsed)

    return mean_latency, p95_latency, throughput


def get_model_size_mb(model_path: Path) -> float:
    """Return model file size in MB."""
    if not model_path.exists():
        return 0.0

    return model_path.stat().st_size / (1024 * 1024)


def is_better_model(
    current_metrics: dict[str, float | int],
    best_metrics: dict[str, float | int] | None,
    monitor_metric: str,
    tie_breaker_metric: str,
) -> bool:
    """Return whether current validation metrics beat the best metrics so far."""
    if best_metrics is None:
        return True

    current_primary = float(current_metrics.get(monitor_metric, 0.0))
    best_primary = float(best_metrics.get(monitor_metric, 0.0))

    if current_primary > best_primary:
        return True

    if current_primary == best_primary:
        current_tie = float(current_metrics.get(tie_breaker_metric, 0.0))
        best_tie = float(best_metrics.get(tie_breaker_metric, 0.0))
        return current_tie > best_tie

    return False


def main() -> int:
    """Train/evaluate one category-specific ResNet18 model."""
    parser = argparse.ArgumentParser(description="Train ResNet18.")
    parser.add_argument("--category", required=True, help="MVTec category.")
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/resnet18_benchmark.yaml")

    category = args.category
    seed = int(config["split"]["random_seed"])
    set_seed(seed)

    split_dir = PROJECT_ROOT / config["outputs"]["split_dir"] / category
    model_dir = PROJECT_ROOT / config["outputs"]["base_model_dir"] / category
    raw_metrics_dir = PROJECT_ROOT / config["outputs"]["raw_metrics_dir"]

    model_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train.csv"
    validation_csv = split_dir / "validation.csv"
    eval_csv = split_dir / "eval.csv"

    if not train_csv.exists() or not validation_csv.exists() or not eval_csv.exists():
        raise FileNotFoundError(
            f"Missing split files for {category}. " "Run scripts/create_resnet18_splits.py first."
        )

    train_df = pd.read_csv(train_csv)
    validation_df = pd.read_csv(validation_csv)
    eval_df = pd.read_csv(eval_csv)

    device = get_device(str(config["training"]["device"]))
    logger.info("Category: %s", category)
    logger.info("Device: %s", device)

    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    num_workers = int(config["training"]["num_workers"])
    pretrained = bool(config["training"]["pretrained"])
    monitor_metric = str(config["training"]["monitor_metric"])
    tie_breaker_metric = str(config["training"]["tie_breaker_metric"])

    train_loader = build_dataloader(
        dataframe=train_df,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=True,
    )
    validation_loader = build_dataloader(
        dataframe=validation_df,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=False,
    )
    eval_loader = build_dataloader(
        dataframe=eval_df,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=False,
    )

    model = build_resnet18_binary_classifier(pretrained=pretrained).to(device)

    class_weights = compute_class_weights(train_df=train_df, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(config["scheduler"]["mode"]),
        factor=float(config["scheduler"]["factor"]),
        patience=int(config["scheduler"]["patience"]),
        min_lr=float(config["scheduler"]["min_lr"]),
    )

    best_model_path = model_dir / "best_model.pt"
    final_model_path = model_dir / "final_model.pt"

    best_validation_metrics: dict[str, float | int] | None = None
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
    # mlflow.set_experiment(str(config["mlflow"]["experiment_name"]))
    setup_mlflow(
        str(config["mlflow"]["tracking_uri"]),
        str(config["mlflow"]["experiment_name"]),
    )

    with mlflow.start_run(run_name=f"resnet18_benchmark_{category}"):
        mlflow.log_params(
            {
                "category": category,
                "model_family": "cnn",
                "model_name": "resnet18",
                "seed": seed,
                "image_size": image_size,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "early_stopping_patience": patience,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "pretrained": pretrained,
                "monitor_metric": monitor_metric,
                "tie_breaker_metric": tie_breaker_metric,
                "train_total": len(train_df),
                "validation_total": len(validation_df),
                "eval_total": len(eval_df),
                "defect_train_fraction": float(config["split"]["defect_train_fraction"]),
                "normal_train_fraction": float(config["split"]["normal_train_fraction"]),
            }
        )

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            validation_loss, validation_metrics, _, _, _ = evaluate(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
            )

            scheduler.step(float(validation_metrics.get(monitor_metric, 0.0)))

            logger.info(
                "%s | epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_recall=%.4f | val_f1=%.4f | val_auroc=%.4f",
                category,
                epoch,
                max_epochs,
                train_loss,
                validation_loss,
                validation_metrics["recall"],
                validation_metrics["f1"],
                validation_metrics["auroc"],
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("validation_loss", validation_loss, step=epoch)

            for metric_name, metric_value in validation_metrics.items():
                mlflow.log_metric(f"validation_{metric_name}", float(metric_value), step=epoch)

            if is_better_model(
                current_metrics=validation_metrics,
                best_metrics=best_validation_metrics,
                monitor_metric=monitor_metric,
                tie_breaker_metric=tie_breaker_metric,
            ):
                best_validation_metrics = validation_metrics
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "category": category,
                        "seed": seed,
                        "image_size": image_size,
                        "validation_metrics": validation_metrics,
                        "config": config,
                    },
                    best_model_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered for %s at epoch %d.", category, epoch)
                break

        training_time_seconds = time.perf_counter() - training_start

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "category": category,
                "seed": seed,
                "image_size": image_size,
                "config": config,
            },
            final_model_path,
        )

        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        eval_loss, eval_metrics, y_true, y_pred, y_score = evaluate(
            model=model,
            dataloader=eval_loader,
            criterion=criterion,
            device=device,
        )

        latency_mean, latency_p95, throughput = benchmark_latency(
            model=model,
            dataloader=eval_loader,
            device=device,
        )

        model_size_mb = get_model_size_mb(best_model_path)

        benchmark_result = BenchmarkResult(
            run_id=f"cnn_resnet18_{category}_seed{seed}",
            model_family="cnn",
            model_name="resnet18",
            category=category,
            seed=seed,
            training_mode="supervised_limited_defect",
            few_shot_k=None,
            classification_auroc=float(eval_metrics["auroc"]),
            classification_f1=float(eval_metrics["f1"]),
            precision=float(eval_metrics["precision"]),
            recall=float(eval_metrics["recall"]),
            false_positive_rate=float(eval_metrics["false_positive_rate"]),
            false_negative_rate=float(eval_metrics["false_negative_rate"]),
            false_negative_count=int(eval_metrics["false_negative"]),
            false_positive_count=int(eval_metrics["false_positive"]),
            segmentation_auroc=None,
            segmentation_f1=None,
            segmentation_aupro=None,
            training_time_seconds=float(training_time_seconds),
            inference_latency_ms_mean=latency_mean,
            inference_latency_ms_p95=latency_p95,
            throughput_images_per_second=throughput,
            model_size_mb=model_size_mb,
            normal_train_fraction=float(config["split"]["normal_train_fraction"]),
            defect_train_fraction=float(config["split"]["defect_train_fraction"]),
            model_artifact_path=str(best_model_path.relative_to(PROJECT_ROOT)),
            report_artifact_path=str(raw_metrics_dir.relative_to(PROJECT_ROOT)),
            notes="Category-specific supervised ResNet18 baseline. Segmentation metrics are not applicable.",
        )

        result_payload = benchmark_result.to_dict()
        result_payload["eval_loss"] = float(eval_loss)
        result_payload["best_validation_metrics"] = best_validation_metrics
        result_payload["raw_predictions"] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
        }

        metrics_path = raw_metrics_dir / f"{category}_resnet18_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(result_payload, file, indent=2)

        for metric_name, metric_value in benchmark_result.to_dict().items():
            if isinstance(metric_value, int | float):
                mlflow.log_metric(metric_name, float(metric_value))

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(best_model_path))

        logger.info("Saved best model: %s", best_model_path)
        logger.info("Saved metrics: %s", metrics_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
