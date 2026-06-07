from __future__ import annotations

from pathlib import Path

import yaml

from drift.trigger_retraining import create_trigger_payload


def test_retraining_config_schema() -> None:
    config = yaml.safe_load(Path("drift/retraining_config.yaml").read_text(encoding="utf-8"))

    assert set(config) == {"inputs", "outputs", "selection", "policy"}
    assert config["inputs"]["drift_report_json"].endswith("drift_report.json")
    assert config["outputs"]["retraining_manifest_csv"].endswith(".csv")
    assert config["outputs"]["retraining_trigger_json"].endswith(".json")
    assert config["selection"]["max_normal_images"] > 0
    assert config["selection"]["max_defective_images"] > 0
    assert config["selection"]["include_all_if_small"] is True
    assert config["policy"]["require_human_labels"] is True


def test_create_trigger_payload_requires_human_review() -> None:
    drift_report = {
        "retraining_recommended": True,
        "decision_reason": "Drift detected; create retraining candidate manifest.",
        "baseline_profile": "baseline",
        "production_profile": "production",
        "production_image_count": 150,
        "minimum_required_production_images": 100,
        "drifted_feature_count": 2,
        "defect_rate_drift": False,
    }
    config = {"policy": {"require_human_labels": True}}

    payload = create_trigger_payload(drift_report=drift_report, config=config)

    assert payload["retraining_required"] is True
    assert payload["requires_human_label_review"] is True
    assert "quality engineer" in payload["label_policy"]
