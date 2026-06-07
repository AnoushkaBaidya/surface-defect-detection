from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class ResNet18OnnxService:
    """
    Production inference wrapper for ResNet18 ONNX Runtime.

    The ONNX session is loaded once at startup and reused across requests.
    """

    def __init__(self, onnx_path: Path) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_score(self, image_tensor: np.ndarray) -> float:
        logits = self.session.run(
            [self.output_name],
            {self.input_name: image_tensor},
        )[0]

        probabilities = softmax(logits[0])
        defect_score = float(probabilities[1])

        return defect_score


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
