from __future__ import annotations

import argparse

from scripts.build_dataset_manifest import run as build_dataset_manifest
from scripts.create_retained_sample_split import run as create_retained_sample_split
from scripts.evaluate_resnet18 import run as evaluate_resnet18
from scripts.generate_results_manifest import run as generate_results_manifest
from scripts.tune_thresholds import run as tune_thresholds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-saved-predictions",
        action="store_true",
        help=(
            "Reuse committed saved predictions and regenerate threshold/report artifacts "
            "from them."
        ),
    )
    args = parser.parse_args()

    if args.from_saved_predictions:
        tune_thresholds()
        evaluate_resnet18(from_saved_predictions=True)
    else:
        build_dataset_manifest()
        create_retained_sample_split()
        evaluate_resnet18(from_saved_predictions=False)
        tune_thresholds()
        evaluate_resnet18(from_saved_predictions=True)
    generate_results_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
