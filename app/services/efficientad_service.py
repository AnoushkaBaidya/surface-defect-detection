from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


class EfficientADService:
    """
    EfficientAD safety-gate service.

    Important:
    The model returns a raw anomaly score. The operating threshold was selected
    on normalized scores, so the raw score must be normalized using the same
    min/max calibration used during offline evaluation.
    """

    def __init__(self, checkpoint_dir: Path, calibration_path: Path, input_size: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        self.raw_min, self.raw_max = self._load_calibration(calibration_path)

        self.model = self._load_model(checkpoint_dir)
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def dependency_summary() -> dict[str, str]:
        return {
            "runtime": "offline_only",
            "required_packages": "torch, torchvision, anomalib",
            "supported_workflows": "checkpoint validation, ONNX parity checks, score calibration",
        }

    def _load_calibration(self, calibration_path: Path) -> tuple[float, float]:
        if not calibration_path.exists():
            raise FileNotFoundError(f"Missing EfficientAD calibration file: {calibration_path}")

        payload = json.loads(calibration_path.read_text(encoding="utf-8"))

        raw_min = float(payload["raw_min"])
        raw_max = float(payload["raw_max"])

        if raw_max <= raw_min:
            raise ValueError(f"Invalid calibration range: raw_min={raw_min}, raw_max={raw_max}")

        return raw_min, raw_max

    def _load_model(self, checkpoint_dir: Path):
        from anomalib.models import EfficientAd

        checkpoint_paths = sorted(checkpoint_dir.rglob("*.ckpt"))

        if not checkpoint_paths:
            raise FileNotFoundError(f"No EfficientAD checkpoint found under: {checkpoint_dir}")

        checkpoint_path = checkpoint_paths[-1]

        # Safe because this checkpoint was produced locally by this project.
        model = EfficientAd.load_from_checkpoint(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )

        return model

    def normalize_score(self, raw_score: float) -> float:
        normalized = (raw_score - self.raw_min) / (self.raw_max - self.raw_min)

        # Clamp for safety if future scores fall outside calibration range.
        return max(0.0, min(1.0, float(normalized)))

    @torch.no_grad()
    def predict_scores_from_pil(self, image: Image.Image) -> dict[str, float]:
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        output = self.model(image_tensor)

        pred_score = getattr(output, "pred_score", None)

        if pred_score is None and isinstance(output, dict):
            pred_score = output.get("pred_score")

        if pred_score is None:
            anomaly_map = getattr(output, "anomaly_map", None)

            if anomaly_map is None and isinstance(output, dict):
                anomaly_map = output.get("anomaly_map")

            if anomaly_map is None:
                raise RuntimeError("Could not extract EfficientAD pred_score or anomaly_map.")

            pred_score = anomaly_map.flatten(start_dim=1).max(dim=1).values

        raw_score = float(pred_score.detach().cpu().numpy().reshape(-1)[0])
        normalized_score = self.normalize_score(raw_score)

        return {
            "raw_score": raw_score,
            "normalized_score": normalized_score,
        }
