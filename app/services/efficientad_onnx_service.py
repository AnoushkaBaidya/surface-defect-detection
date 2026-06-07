from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class EfficientADOnnxService:
    """
    EfficientAD ONNX Runtime safety-gate service.

    The ONNX model returns a raw anomaly score. We normalize it with the
    min/max calibration saved from the offline model-selection evaluation.
    """

    def __init__(self, onnx_path: Path, calibration_path: Path) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"EfficientAD ONNX model not found: {onnx_path}")

        if not calibration_path.exists():
            raise FileNotFoundError(f"EfficientAD calibration file not found: {calibration_path}")

        self.raw_min, self.raw_max = self._load_calibration(calibration_path)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = 8
        session_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _load_calibration(self, calibration_path: Path) -> tuple[float, float]:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))

        raw_min = float(payload["raw_min"])
        raw_max = float(payload["raw_max"])

        if raw_max <= raw_min:
            raise ValueError(
                f"Invalid EfficientAD calibration range: raw_min={raw_min}, raw_max={raw_max}"
            )

        return raw_min, raw_max

    def normalize_score(self, raw_score: float) -> float:
        normalized = (raw_score - self.raw_min) / (self.raw_max - self.raw_min)
        return max(0.0, min(1.0, float(normalized)))

    def predict_scores(self, image_tensor: np.ndarray) -> dict[str, float]:
        raw_score_array = self.session.run(
            [self.output_name],
            {self.input_name: image_tensor},
        )[0]

        raw_score = float(raw_score_array.reshape(-1)[0])
        normalized_score = self.normalize_score(raw_score)

        return {
            "raw_score": raw_score,
            "normalized_score": normalized_score,
        }
