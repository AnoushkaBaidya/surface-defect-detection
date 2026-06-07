"""
Retraining Trigger Logic

The retraining trigger should not start model retraining immediately.
Instead, it should prepare a reviewable retraining package.

Workflow:
1. Read the latest drift report.
2. Check configured drift and performance thresholds.
3. Create a retraining candidate manifest.
4. Write retraining_required.json.
5. Log the reason retraining was requested.

Trigger Conditions:
If drift is detected in brightness, contrast, or sharpness,
OR the reviewed defect rate shifts significantly,
OR manual review labels indicate degraded model performance,
THEN create a retraining package.

Expected Output:
- retraining_required.json
- candidate image manifest
- trigger reason
- timestamp
- drift metrics summary

Goal:
Separate retraining decision logic from actual retraining execution.
This is more realistic because retraining usually
requires review, approval, scheduling, and reproducibility checks before
a new model is trained or promoted.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path, label: int, limit: int) -> list[dict]:
    images = [path for path in sorted(folder.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]

    return [
        {
            "image_path": str(path),
            "label": label,
            "label_source": "requires_human_review",
        }
        for path in images[:limit]
    ]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_drift_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create_trigger_payload(drift_report: dict, config: dict) -> dict:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "retraining_required": bool(drift_report["retraining_recommended"]),
        "reason": drift_report["decision_reason"],
        "baseline_profile": drift_report["baseline_profile"],
        "production_profile": drift_report["production_profile"],
        "production_image_count": drift_report["production_image_count"],
        "minimum_required_production_images": drift_report.get(
            "minimum_required_production_images"
        ),
        "drifted_feature_count": drift_report["drifted_feature_count"],
        "defect_rate_drift": drift_report["defect_rate_drift"],
        "requires_human_label_review": config["policy"]["require_human_labels"],
        "label_policy": (
            "Candidate images must be reviewed by an operator or quality engineer "
            "before any retraining job is approved."
        ),
    }


def create_candidate_manifest(config: dict) -> list[dict]:
    normal_folder = Path(config["inputs"]["production_normal_dir"])
    defective_folder = Path(config["inputs"]["production_defective_dir"])

    normal_paths = [
        path for path in sorted(normal_folder.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    defective_paths = [
        path
        for path in sorted(defective_folder.rglob("*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    include_all_if_small = bool(config["selection"].get("include_all_if_small", False))

    normal_limit = int(config["selection"]["max_normal_images"])
    defective_limit = int(config["selection"]["max_defective_images"])

    if include_all_if_small and len(normal_paths) <= normal_limit:
        normal_limit = len(normal_paths)

    if include_all_if_small and len(defective_paths) <= defective_limit:
        defective_limit = len(defective_paths)

    rows: list[dict] = []

    rows.extend(
        list_images(
            normal_folder,
            label=0,
            limit=normal_limit,
        )
    )

    rows.extend(
        list_images(
            defective_folder,
            label=1,
            limit=defective_limit,
        )
    )

    return rows


def write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "label", "label_source"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    config = load_yaml(Path("drift/retraining_config.yaml"))

    drift_report = load_drift_report(Path(config["inputs"]["drift_report_json"]))
    trigger = create_trigger_payload(drift_report=drift_report, config=config)

    output_trigger = Path(config["outputs"]["retraining_trigger_json"])
    output_trigger.parent.mkdir(parents=True, exist_ok=True)
    output_trigger.write_text(json.dumps(trigger, indent=2), encoding="utf-8")

    if not trigger["retraining_required"]:
        print("Retraining not required.")
        print(json.dumps(trigger, indent=2))
        return 0

    rows = create_candidate_manifest(config)
    manifest_path = Path(config["outputs"]["retraining_manifest_csv"])
    write_manifest(manifest_path=manifest_path, rows=rows)

    print("Retraining triggered.")
    print(json.dumps(trigger, indent=2))
    print(f"Retraining manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
