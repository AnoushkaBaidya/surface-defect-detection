from __future__ import annotations

from drift.detect_drift import create_drift_report, relative_change


def test_relative_change() -> None:
    assert relative_change(100.0, 120.0) == 0.2
    assert relative_change(100.0, 80.0) == -0.2


def test_create_drift_report_respects_minimum_sample_size() -> None:
    baseline = {
        "profile_name": "baseline",
        "image_count": 100,
        "defect_rate": 0.1,
        "features": {
            "brightness_mean": {"mean": 100.0},
            "contrast_std": {"mean": 20.0},
            "sharpness_laplacian_var": {"mean": 50.0},
        },
        "rows": [
            {
                "brightness_mean": 100.0,
                "contrast_std": 20.0,
                "sharpness_laplacian_var": 50.0,
            }
            for _ in range(100)
        ],
    }
    production = {
        "profile_name": "production",
        "image_count": 2,
        "defect_rate": 0.9,
        "features": {
            "brightness_mean": {"mean": 50.0},
            "contrast_std": {"mean": 5.0},
            "sharpness_laplacian_var": {"mean": 10.0},
        },
        "rows": [
            {
                "brightness_mean": 50.0,
                "contrast_std": 5.0,
                "sharpness_laplacian_var": 10.0,
            }
            for _ in range(2)
        ],
    }
    config = {
        "thresholds": {
            "brightness_relative_change_warning": 0.15,
            "contrast_relative_change_warning": 0.15,
            "sharpness_relative_change_warning": 0.20,
            "defect_rate_absolute_change_warning": 0.10,
            "ks_pvalue_warning": 0.05,
        },
        "retraining_trigger": {
            "min_drifted_features_for_warning": 2,
            "min_total_images": 100,
        },
    }

    report = create_drift_report(baseline=baseline, production=production, config=config)

    assert report["drifted_feature_count"] == 3
    assert report["defect_rate_drift"] is True
    assert report["enough_production_data"] is False
    assert report["retraining_recommended"] is False
