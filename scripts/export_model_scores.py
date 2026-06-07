"""
Export per-image prediction scores for hybrid fusion.

This reruns prediction for PatchCore, FastFlow, or WinCLIP and saves:

artifacts/benchmarks/hybrid/raw_scores/<model>/<category>.csv

Why this exists
---------------
Hybrid fusion requires per-image scores, not aggregate AUROC/F1 values.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from defect_detection.evaluation.anomalib_metrics import (
    extract_prediction_scores_table,
)  # noqa: E402
from defect_detection.utils.config import load_yaml_config  # noqa: E402
from defect_detection.utils.logging import get_logger  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


logger = get_logger(__name__)


def resolve_accelerator() -> str:
    """Use GPU if available, otherwise CPU."""
    return "gpu" if torch.cuda.is_available() else "cpu"


def build_model(model_name: str, category: str, k_shot: int | None):
    """Build model with conservative Anomalib-compatible defaults."""
    if model_name == "patchcore":
        from anomalib.models import Patchcore

        try:
            return Patchcore(
                backbone="wide_resnet50_2",
                layers=("layer2", "layer3"),
                coreset_sampling_ratio=0.1,
                num_neighbors=9,
            )
        except TypeError:
            return Patchcore()

    if model_name == "fastflow":
        from anomalib.models import Fastflow

        try:
            return Fastflow(
                backbone="resnet18",
                pre_trained=True,
                flow_steps=8,
                conv3x3_only=False,
                hidden_ratio=1.0,
            )
        except TypeError:
            return Fastflow()

    if model_name == "winclip":
        try:
            from anomalib.models.image import WinClip
        except ImportError:
            from anomalib.models import WinClip

        class_name = category.replace("_", " ")
        k_value = int(k_shot or 0)

        try:
            return WinClip(class_name=class_name, k_shot=k_value, scales=(2, 3))
        except TypeError:
            try:
                return WinClip(class_name=class_name, k_shot=k_value)
            except TypeError:
                return WinClip(k_shot=k_value)

    raise ValueError(f"Unsupported model: {model_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export hybrid benchmark per-image scores.")
    parser.add_argument("--model", required=True, choices=["patchcore", "fastflow", "winclip"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--k-shot", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/hybrid_score_fusion.yaml")

    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    dataset_root = PROJECT_ROOT / "data/raw/mvtec_ad"
    accelerator = resolve_accelerator()

    datamodule = MVTecAD(
        root=dataset_root,
        category=args.category,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )

    model = build_model(args.model, args.category, args.k_shot)

    try:
        engine = Engine(accelerator=accelerator, devices=1, max_epochs=1)
    except TypeError:
        engine = Engine()

    start = time.perf_counter()

    if args.model in {"patchcore", "fastflow"}:
        engine.fit(model=model, datamodule=datamodule)

    predictions = engine.predict(model=model, datamodule=datamodule)
    elapsed = time.perf_counter() - start

    rows = extract_prediction_scores_table(predictions)

    if not rows:
        raise RuntimeError(
            "Could not extract per-image scores from predictions. "
            "We need to inspect the prediction object structure."
        )

    output_root = PROJECT_ROOT / config["outputs"]["raw_scores_dir"] / args.model
    if args.model == "winclip":
        output_root = output_root / f"k{args.k_shot}"

    output_root.mkdir(parents=True, exist_ok=True)

    output_path = output_root / f"{args.category}.csv"

    df = pd.DataFrame(rows)
    df["model_name"] = args.model
    df["category"] = args.category
    df["few_shot_k"] = args.k_shot
    df["export_elapsed_seconds"] = elapsed

    df.to_csv(output_path, index=False)

    logger.info("Saved score file: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
