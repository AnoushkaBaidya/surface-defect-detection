from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import (
    IMAGE_EXTENSIONS,
    REPO_ROOT,
    label_to_int,
    load_yaml,
    write_json,
    write_rows,
)


def build_manifest_rows(config: dict) -> list[dict]:
    dataset_root = REPO_ROOT / config["dataset_root"]
    rows: list[dict] = []

    for dataset in config["datasets"]:
        dataset_name = dataset["name"]
        capture_batch = dataset["capture_batch"]
        folder = dataset_root / dataset_name

        for label in ("normal", "defective"):
            label_dir = folder / label
            for image_path in sorted(label_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                relative_path = image_path.relative_to(REPO_ROOT)
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "capture_batch": capture_batch,
                        "label": label,
                        "label_id": label_to_int(label),
                        "relative_path": str(relative_path).replace("\\", "/"),
                        "is_private": False,
                        "is_simulated": dataset.get("is_simulated", False),
                    }
                )

    return rows


def build_summary(rows: list[dict]) -> dict:
    by_dataset: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = by_dataset.setdefault(
            row["dataset_name"],
            {"normal": 0, "defective": 0, "total": 0},
        )
        counts[row["label"]] += 1
        counts["total"] += 1
    return by_dataset


def run(config_path: str = "configs/pipeline/dataset_retained_sample.yaml") -> Path:
    config = load_yaml(REPO_ROOT / config_path)
    manifest_rows = build_manifest_rows(config)

    output_csv = REPO_ROOT / config["outputs"]["manifest_csv"]
    write_rows(
        output_csv,
        manifest_rows,
        [
            "dataset_name",
            "capture_batch",
            "label",
            "label_id",
            "relative_path",
            "is_private",
            "is_simulated",
        ],
    )

    summary = {
        "dataset_root": config["dataset_root"],
        "row_count": len(manifest_rows),
        "datasets": build_summary(manifest_rows),
    }
    write_json(REPO_ROOT / config["outputs"]["manifest_summary_json"], summary)
    return output_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pipeline/dataset_retained_sample.yaml",
        help="Dataset contract config used to build the retained-sample manifest.",
    )
    args = parser.parse_args()

    output_csv = run(args.config)

    print(f"Saved manifest: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
