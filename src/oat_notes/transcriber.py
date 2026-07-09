"""Transcription backends behind a common interface.

Callers only ever see ``create_transcriber(config)``: faster-whisper on CPU
by default, or OpenVINO GenAI (``--backend openvino``) to offload onto the
Intel NPU/GPU and keep CPU cores free during meetings.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

from .config import Config
from .types import AudioChunk, TranscriptSegment


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, chunk: AudioChunk) -> TranscriptSegment: ...


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, config: Config) -> None:
        from faster_whisper import WhisperModel

        self._config = config
        self._model = WhisperModel(
            config.model_name, device="cpu", compute_type=config.compute_type
        )

    def transcribe(self, chunk: AudioChunk) -> TranscriptSegment:
        segments, _ = self._model.transcribe(
            chunk.samples,
            language=self._config.language,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptSegment(
            text=text, channel=chunk.channel, start=chunk.start, end=chunk.end
        )


class OpenVinoTranscriber(Transcriber):
    """OpenVINO GenAI WhisperPipeline — offloads inference to NPU or GPU.

    The default model is a pre-converted int8 repo from the OpenVINO HF org,
    so no torch/optimum conversion toolchain is needed. If the requested
    device can't take the model, falls back to OpenVINO on CPU (plan B).
    """

    def __init__(self, config: Config) -> None:
        try:
            import openvino_genai
        except ImportError as error:
            raise RuntimeError(
                "The openvino backend is an optional extra — install it with:"
                " uv sync --extra openvino"
            ) from error

        model_dir = self._resolve_model(config.openvino_model)
        self.device = config.openvino_device
        try:
            self._pipeline = openvino_genai.WhisperPipeline(
                model_dir, device=self.device
            )
        except Exception as error:
            if self.device == "CPU":
                raise
            print(
                f"warning: OpenVINO {self.device} rejected the model"
                f" ({error}); falling back to CPU",
                file=sys.stderr,
            )
            self.device = "CPU"
            self._pipeline = openvino_genai.WhisperPipeline(model_dir, device="CPU")

    @staticmethod
    def _resolve_model(source: str) -> str:
        """``source`` is a local directory or a HuggingFace repo id.

        Repos download to %LOCALAPPDATA%/oat-notes/models as plain files:
        the default HF cache layout needs symlinks, which Windows only
        allows with Developer Mode enabled.
        """
        if Path(source).is_dir():
            return source
        import os

        from huggingface_hub import snapshot_download

        target = (
            Path(os.environ["LOCALAPPDATA"])
            / "oat-notes"
            / "models"
            / source.replace("/", "--")
        )
        return snapshot_download(source, local_dir=target)

    def transcribe(self, chunk: AudioChunk) -> TranscriptSegment:
        result = self._pipeline.generate(chunk.samples)
        text = " ".join(piece.strip() for piece in result.texts).strip()
        return TranscriptSegment(
            text=text, channel=chunk.channel, start=chunk.start, end=chunk.end
        )


def create_transcriber(config: Config) -> Transcriber:
    if config.backend == "faster_whisper":
        return FasterWhisperTranscriber(config)
    if config.backend == "openvino":
        return OpenVinoTranscriber(config)
    raise ValueError(f"Unknown transcription backend: {config.backend!r}")
