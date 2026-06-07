from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from scripts.common import REPO_ROOT, load_rows, load_yaml, write_rows


def train_validation_split(
    rows: list[dict], validation_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    grouped: dict[int, list[dict]] = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label_id"])].append(row)

    rng = random.Random(seed)
    train_rows: list[dict] = []
    validation_rows: list[dict] = []

    for _label_id, label_rows in grouped.items():
        rng.shuffle(label_rows)
        validation_count = max(1, int(round(len(label_rows) * validation_fraction)))
        validation_rows.extend(label_rows[:validation_count])
        train_rows.extend(label_rows[validation_count:])

    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    return train_rows, validation_rows


def split_summary(rows: list[dict]) -> dict[str, int]:
    labels = Counter(row["label"] for row in rows)
    return {
        "normal": labels.get("normal", 0),
        "defective": labels.get("defective", 0),
        "total": len(rows),
    }


def run(config_path: str = "configs/pipeline/train_resnet18_retained_sample.yaml") -> Path:
    config = load_yaml(REPO_ROOT / config_path)
    manifest_rows = load_rows(REPO_ROOT / config["dataset"]["manifest_csv"])

    baseline_rows = [row for row in manifest_rows if row["dataset_name"] == "baseline_oct14"]
    production_rows = [row for row in manifest_rows if row["dataset_name"] == "production_nov8"]

    train_rows, validation_rows = train_validation_split(
        baseline_rows,
        validation_fraction=float(config["split"]["validation_fraction"]),
        seed=int(config["split"]["seed"]),
    )

    output_dir = REPO_ROOT / config["outputs"]["split_dir"]
    write_rows(output_dir / "train.csv", train_rows, list(train_rows[0].keys()))
    write_rows(output_dir / "validation.csv", validation_rows, list(validation_rows[0].keys()))
    write_rows(output_dir / "test.csv", production_rows, list(production_rows[0].keys()))

    summary_rows = [
        {"split": "train", **split_summary(train_rows)},
        {"split": "validation", **split_summary(validation_rows)},
        {"split": "test", **split_summary(production_rows)},
    ]
    write_rows(
        output_dir / "split_summary.csv", summary_rows, ["split", "normal", "defective", "total"]
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pipeline/train_resnet18_retained_sample.yaml",
    )
    args = parser.parse_args()

    output_dir = run(args.config)

    print(f"Saved split CSVs under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
