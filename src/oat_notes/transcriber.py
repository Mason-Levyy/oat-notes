"""Transcription backends behind ``create_transcriber``: faster-whisper on
CPU, or OpenVINO GenAI offloading to the Intel NPU/GPU."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import Config
from .models import (
    ensure_tls_trust,
    load_on_device_or_fall_back,
    openvino_cache_dir,
    resolve_openvino_model,
)
from .repetition import collapse_repeated_runs
from .types import AudioChunk, TranscriptSegment

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionHints:
    """Text Whisper sees before the audio. ``prompt`` is what was just said,
    so a fragment continues the sentence instead of starting a new one;
    ``hotwords`` are spellings worth biasing towards for every chunk."""

    prompt: str | None = None
    hotwords: str | None = None


NO_HINTS = TranscriptionHints()


class Transcriber(ABC):
    """Implementations wrap a single model handle and are called from more
    than one thread — the meeting worker and dictation can overlap — so each
    serializes ``transcribe`` on its own lock."""

    @abstractmethod
    def transcribe(
        self, chunk: AudioChunk, hints: TranscriptionHints = NO_HINTS
    ) -> TranscriptSegment: ...


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
            log.info("downloading %s", config.model_name)
            ensure_tls_trust()
            self._model = WhisperModel(
                config.model_name, device="cpu", compute_type=config.compute_type
            )

    def transcribe(
        self, chunk: AudioChunk, hints: TranscriptionHints = NO_HINTS
    ) -> TranscriptSegment:
        with self._lock:
            segments, _ = self._model.transcribe(
                chunk.samples,
                language=self._config.language,
                beam_size=1,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=False,
                repetition_penalty=self._config.repetition_penalty,
                compression_ratio_threshold=self._config.compression_ratio_threshold,
                initial_prompt=hints.prompt,
                hotwords=hints.hotwords,
            )
            spoken = [segment.text.strip() for segment in segments]
        text = collapse_repeated_runs(" ".join(spoken).strip())
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
        self._lock = threading.Lock()
        cache_dir = str(openvino_cache_dir())
        self._pipeline, self.device = load_on_device_or_fall_back(
            lambda device: openvino_genai.WhisperPipeline(
                model_dir, device=device, CACHE_DIR=cache_dir
            ),
            config.openvino_device,
            "transcription model",
        )

    def transcribe(
        self, chunk: AudioChunk, hints: TranscriptionHints = NO_HINTS
    ) -> TranscriptSegment:
        with self._lock:
            result = self._pipeline.generate(
                chunk.samples, **_openvino_hint_kwargs(hints)
            )
        text = collapse_repeated_runs(
            " ".join(piece.strip() for piece in result.texts).strip()
        )
        return TranscriptSegment(
            text=text, channel=chunk.channel, start=chunk.start, end=chunk.end
        )


def _openvino_hint_kwargs(hints: TranscriptionHints) -> dict[str, str]:
    """WhisperPipeline rejects ``None`` for these, so only pass what is set."""
    kwargs = {}
    if hints.prompt:
        kwargs["initial_prompt"] = hints.prompt
    if hints.hotwords:
        kwargs["hotwords"] = hints.hotwords
    return kwargs


def create_transcriber(config: Config) -> Transcriber:
    if config.backend == "faster_whisper":
        return FasterWhisperTranscriber(config)
    if config.backend == "openvino":
        return OpenVinoTranscriber(config)
    raise ValueError(f"Unknown transcription backend: {config.backend!r}")
