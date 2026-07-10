"""Streaming Silero VAD over the ONNX model bundled with faster-whisper,
whose own wrapper is batch-only: this session carries the LSTM state and the
64-sample context tail across calls, one 512-sample window at a time."""

import os

import numpy as np
import onnxruntime
from faster_whisper.utils import get_assets_path

WINDOW_SAMPLES = 512
_CONTEXT_SAMPLES = 64
_MODEL_FILENAME = "silero_vad_v6.onnx"


class SileroVad:
    window_samples = WINDOW_SAMPLES

    def __init__(self) -> None:
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            os.path.join(get_assets_path(), _MODEL_FILENAME),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)

    def speech_probability(self, window: np.ndarray) -> float:
        if window.shape != (WINDOW_SAMPLES,):
            raise ValueError(
                f"expected exactly {WINDOW_SAMPLES} samples, got {window.shape}"
            )
        window = window.astype(np.float32, copy=False)
        model_input = np.concatenate([self._context, window])[np.newaxis, :]
        probs, self._h, self._c = self._session.run(
            None, {"input": model_input, "h": self._h, "c": self._c}
        )
        self._context = window[-_CONTEXT_SAMPLES:]
        return float(probs[0])
