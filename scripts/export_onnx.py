from __future__ import annotations

import argparse
import json

import torch

from scripts.common import (
    REPO_ROOT,
    build_resnet18_binary_classifier,
    file_sha256,
    load_torch_checkpoint,
    load_yaml,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pipeline/export_resnet18_retained_sample.yaml",
    )
    args = parser.parse_args()

    config = load_yaml(REPO_ROOT / args.config)
    checkpoint_path = REPO_ROOT / config["input"]["checkpoint_path"]
    onnx_path = REPO_ROOT / config["output"]["onnx_path"]
    input_size = int(config["model"]["input_size"])

    device = torch.device("cpu")
    checkpoint = load_torch_checkpoint(checkpoint_path, device)

    model = build_resnet18_binary_classifier().to(device)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    lineage = {
        "source_checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "exported_onnx_path": str(onnx_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "onnx_sha256": file_sha256(onnx_path),
        "checkpoint_metadata": checkpoint.metadata,
    }
    write_json(REPO_ROOT / config["output"]["lineage_json"], lineage)
    print(json.dumps(lineage, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
