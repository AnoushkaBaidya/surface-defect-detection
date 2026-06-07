from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.common import (
    REPO_ROOT,
    build_resnet18_binary_classifier,
    confusion_metrics,
    file_sha256,
    load_rows,
    load_yaml,
    make_loader,
    score_distribution_summary,
    set_seed,
    write_json,
)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_score: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * images.size(0)
            y_true.extend(labels.cpu().tolist())
            y_score.extend(probabilities.cpu().tolist())

    mean_loss = total_loss / max(len(loader.dataset), 1)
    return mean_loss, np.array(y_true, dtype=int), np.array(y_score, dtype=float)


def train(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["training"]["seed"]))

    split_dir = REPO_ROOT / config["outputs"]["split_dir"]
    train_rows = load_rows(split_dir / "train.csv")
    validation_rows = load_rows(split_dir / "validation.csv")

    batch_size = int(config["training"]["batch_size"])
    input_size = int(config["model"]["input_size"])
    num_workers = int(config["training"]["num_workers"])

    train_loader = make_loader(
        train_rows,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=True,
    )
    validation_loader = make_loader(
        validation_rows,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=False,
    )

    device = torch.device(config["training"]["device"])
    model = build_resnet18_binary_classifier().to(device)

    class_counts = {0: 0, 1: 0}
    for row in train_rows:
        class_counts[int(row["label_id"])] += 1
    total = sum(class_counts.values())
    class_weights = torch.tensor(
        [
            total / max(class_counts[0] * 2, 1),
            total / max(class_counts[1] * 2, 1),
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_f1 = -1.0
    best_payload: dict[str, Any] | None = None
    best_state_dict: dict[str, Any] | None = None
    no_improve = 0
    history: list[dict[str, Any]] = []
    train_start = time.perf_counter()

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        validation_loss, y_true, y_score = evaluate_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        threshold = float(config["thresholding"]["default_threshold"])
        metrics = confusion_metrics(y_true=y_true, y_score=y_score, threshold=threshold)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(len(train_loader.dataset), 1),
                "validation_loss": validation_loss,
            }
        )
        history.append(metrics)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_payload = metrics
            best_state_dict = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(config["training"]["early_stopping_patience"]):
            break

    training_seconds = time.perf_counter() - train_start
    if best_state_dict is None or best_payload is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    checkpoint_path = REPO_ROOT / config["outputs"]["checkpoint_path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "metadata": {
                "training_seconds": training_seconds,
                "training_config_path": config["metadata"]["config_path"],
                "split_dir": config["outputs"]["split_dir"],
                "validation_score_distribution": score_distribution_summary(
                    [row["f1"] for row in history]
                ),
                "best_epoch": best_payload["epoch"],
            },
        },
        checkpoint_path,
    )

    history_path = REPO_ROOT / config["outputs"]["training_history_json"]
    write_json(history_path, {"history": history})

    summary = {
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_seconds": training_seconds,
        "best_validation_metrics": best_payload,
    }
    write_json(REPO_ROOT / config["outputs"]["training_summary_json"], summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pipeline/train_resnet18_retained_sample.yaml",
    )
    args = parser.parse_args()

    config = load_yaml(REPO_ROOT / args.config)
    config.setdefault("metadata", {})["config_path"] = args.config
    summary = train(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
