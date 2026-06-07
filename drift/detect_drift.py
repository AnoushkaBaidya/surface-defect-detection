from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import ks_2samp


def relative_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0

    return float((new - old) / old)


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_values(profile: dict[str, Any], feature: str) -> list[float]:
    return [float(row[feature]) for row in profile["rows"]]


def compare_feature(
    baseline: dict[str, Any],
    production: dict[str, Any],
    feature: str,
    warning_threshold: float,
) -> dict[str, Any]:
    base_mean = baseline["features"][feature]["mean"]
    prod_mean = production["features"][feature]["mean"]

    rel_change = relative_change(base_mean, prod_mean)

    base_values = feature_values(baseline, feature)
    prod_values = feature_values(production, feature)

    ks_stat, ks_pvalue = ks_2samp(base_values, prod_values)

    return {
        "feature": feature,
        "baseline_mean": base_mean,
        "production_mean": prod_mean,
        "relative_change": rel_change,
        "abs_relative_change": abs(rel_change),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "mean_shift_drift": abs(rel_change) >= warning_threshold,
    }


def create_drift_report(
    baseline: dict[str, Any],
    production: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["thresholds"]

    feature_threshold_map = {
        "brightness_mean": thresholds["brightness_relative_change_warning"],
        "contrast_std": thresholds["contrast_relative_change_warning"],
        "sharpness_laplacian_var": thresholds["sharpness_relative_change_warning"],
    }

    feature_results = []

    for feature, threshold in feature_threshold_map.items():
        result = compare_feature(
            baseline=baseline,
            production=production,
            feature=feature,
            warning_threshold=threshold,
        )
        result["ks_drift"] = result["ks_pvalue"] < thresholds["ks_pvalue_warning"]
        result["drift_detected"] = result["mean_shift_drift"] or result["ks_drift"]
        feature_results.append(result)

    defect_rate_change = production["defect_rate"] - baseline["defect_rate"]
    defect_rate_drift = abs(defect_rate_change) >= thresholds["defect_rate_absolute_change_warning"]

    drifted_features = [result for result in feature_results if result["drift_detected"]]
    min_total_images = int(config["retraining_trigger"]["min_total_images"])
    enough_production_data = production["image_count"] >= min_total_images

    retraining_recommended = enough_production_data and (
        len(drifted_features) >= config["retraining_trigger"]["min_drifted_features_for_warning"]
        or defect_rate_drift
    )

    decision_reason = "No significant drift detected."

    if not enough_production_data:
        decision_reason = (
            "Insufficient production sample size for retraining decision; continue monitoring."
        )
    elif retraining_recommended:
        decision_reason = "Drift detected; create retraining candidate manifest."

    return {
        "baseline_profile": baseline["profile_name"],
        "production_profile": production["profile_name"],
        "baseline_image_count": baseline["image_count"],
        "production_image_count": production["image_count"],
        "minimum_required_production_images": min_total_images,
        "enough_production_data": enough_production_data,
        "baseline_defect_rate": baseline["defect_rate"],
        "production_defect_rate": production["defect_rate"],
        "defect_rate_change": defect_rate_change,
        "defect_rate_drift": defect_rate_drift,
        "feature_results": feature_results,
        "drifted_feature_count": len(drifted_features),
        "retraining_recommended": retraining_recommended,
        "decision_reason": decision_reason,
    }


def main() -> int:
    config = yaml.safe_load(Path("drift/drift_config.yaml").read_text(encoding="utf-8"))

    baseline = load_json(config["outputs"]["baseline_profile_json"])
    production = load_json(config["outputs"]["production_profile_json"])

    report = create_drift_report(baseline=baseline, production=production, config=config)

    output_json = Path(config["outputs"]["drift_report_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    output_md = Path(config["outputs"]["drift_report_md"])
    output_md.write_text(
        create_markdown_report(report),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"Saved drift report: {output_json}")
    print(f"Saved markdown report: {output_md}")

    return 0


def create_markdown_report(report: dict) -> str:
    lines = [
        "# Drift Detection Report",
        "",
        f"Baseline profile: `{report['baseline_profile']}`",
        f"Production profile: `{report['production_profile']}`",
        "",
        "## Summary",
        "",
        f"- Baseline images: `{report['baseline_image_count']}`",
        f"- Production images: `{report['production_image_count']}`",
        f"- Baseline defect rate: `{report['baseline_defect_rate']:.4f}`",
        f"- Production defect rate: `{report['production_defect_rate']:.4f}`",
        f"- Minimum production images required: `{report['minimum_required_production_images']}`",
        f"- Enough production data: `{report['enough_production_data']}`",
        f"- Retraining recommended: `{report['retraining_recommended']}`",
        f"- Decision: {report['decision_reason']}",
        "",
        "## Feature Drift",
        "",
        "| Feature | Baseline Mean | Production Mean | Relative Change | KS p-value | Drift |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for result in report["feature_results"]:
        lines.append(
            f"| {result['feature']} "
            f"| {result['baseline_mean']:.4f} "
            f"| {result['production_mean']:.4f} "
            f"| {result['relative_change']:.4f} "
            f"| {result['ks_pvalue']:.6f} "
            f"| {result['drift_detected']} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
