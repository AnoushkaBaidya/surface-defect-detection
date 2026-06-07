from __future__ import annotations

import argparse
import csv
import json

from scripts.common import REPO_ROOT, load_yaml, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pipeline/evaluate_resnet18_retained_sample.yaml",
    )
    parser.add_argument(
        "--scores-csv",
        required=False,
        help="CSV containing a column named efficientad_score_raw.",
    )
    args = parser.parse_args()

    config = load_yaml(REPO_ROOT / args.config)
    calibration_path = REPO_ROOT / "configs/efficientad_calibration.json"

    if args.scores_csv is None:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        write_json(REPO_ROOT / config["outputs"]["efficientad_calibration_copy_json"], payload)
        print(json.dumps(payload, indent=2))
        return 0

    raw_values: list[float] = []
    with (REPO_ROOT / args.scores_csv).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_values.append(float(row["efficientad_score_raw"]))

    payload = {
        "model_name": "efficientad_safety_gate",
        "score_type": "minmax_normalization",
        "raw_score_column": "efficientad_score_raw",
        "raw_min": min(raw_values),
        "raw_max": max(raw_values),
        "normalized_threshold": 0.30,
        "formula": "normalized_score = (raw_score - raw_min) / (raw_max - raw_min)",
        "source": args.scores_csv,
    }
    write_json(REPO_ROOT / config["outputs"]["efficientad_calibration_copy_json"], payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
