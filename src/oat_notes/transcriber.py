"""Transcription backends behind ``create_transcriber``: faster-whisper on
CPU, or OpenVINO GenAI offloading to the Intel NPU/GPU."""

from __future__ import annotations

import sys
import threading
from abc import ABC, abstractmethod

from .config import Config
from .models import (
    ensure_tls_trust,
    openvino_cache_dir,
    openvino_model_cached,
    resolve_openvino_model,
)
from .types import AudioChunk, TranscriptSegment

__all__ = [
    "Transcriber",
    "FasterWhisperTranscriber",
    "OpenVinoTranscriber",
    "create_transcriber",
    "ensure_tls_trust",
    "openvino_model_cached",
    "resolve_openvino_model",
]


class Transcriber(ABC):
    """Implementations wrap a single model handle and are called from more
    than one thread — the meeting worker and dictation can overlap — so each
    serializes ``transcribe`` on its own lock."""

    @abstractmethod
    def transcribe(self, chunk: AudioChunk) -> TranscriptSegment: ...


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, config: Config) -> None:
        from faster_whisper import WhisperModel

        self._config = config
        self._lock = threading.Lock()
        try:
            self._model = WhisperModel(
                config.model_name,
                device="cpu",
                compute_type=config.compute_type,
                local_files_only=True,
            )
        except Exception:
            if config.offline:
                raise RuntimeError(
                    f"model {config.model_name!r} is not cached locally and"
                    " --offline forbids downloading it"
                ) from None
            print(f"downloading {config.model_name}…", file=sys.stderr)
            ensure_tls_trust()
            self._model = WhisperModel(
                config.model_name, device="cpu", compute_type=config.compute_type
            )

    def transcribe(self, chunk: AudioChunk) -> TranscriptSegment:
        with self._lock:
            segments, _ = self._model.transcribe(
                chunk.samples,
                language=self._config.language,
                beam_size=1,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=False,
            )
            spoken = [segment.text.strip() for segment in segments]
        text = " ".join(spoken).strip()
        return TranscriptSegment(
            text=text, channel=chunk.channel, start=chunk.start, end=chunk.end
        )


class OpenVinoTranscriber(Transcriber):
    """WhisperPipeline on NPU/GPU using pre-converted int8 models (no torch
    toolchain); a device that rejects the model falls back to CPU."""

    def __init__(self, config: Config) -> None:
        try:
            import openvino_genai
        except ImportError as error:
            raise RuntimeError(
                "The openvino backend is an optional extra — install it with:"
                " uv sync --extra openvino"
            ) from error

        model_dir = resolve_openvino_model(config.openvino_model, config.offline)
        self.device = config.openvino_device
        self._lock = threading.Lock()
        cache_dir = openvino_cache_dir()
        try:
            self._pipeline = openvino_genai.WhisperPipeline(
                model_dir, device=self.device, CACHE_DIR=str(cache_dir)
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

    def transcribe(self, chunk: AudioChunk) -> TranscriptSegment:
        with self._lock:
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
