from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from drift.compute_drift_features import build_feature_rows, summarize_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="drift/drift_config.yaml")
    parser.add_argument("--mode", choices=["baseline", "production"], required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    section = config[args.mode]

    normal_dir = Path(section["normal_dir"])
    defective_dir = Path(section["defective_dir"])

    rows = build_feature_rows(normal_dir=normal_dir, defective_dir=defective_dir)
    profile = summarize_profile(rows=rows, profile_name=section["name"]).to_dict()

    output_key = "baseline_profile_json" if args.mode == "baseline" else "production_profile_json"

    output_path = Path(config["outputs"][output_key])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"Saved {args.mode} profile: {output_path}")
    print(json.dumps({k: v for k, v in profile.items() if k != "rows"}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
