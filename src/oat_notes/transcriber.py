"""Transcription backends behind a common interface.

Phase 4 adds an OpenVINO/NPU backend as another subclass; callers only ever
see ``create_transcriber(config)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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


def create_transcriber(config: Config) -> Transcriber:
    if config.backend == "faster_whisper":
        return FasterWhisperTranscriber(config)
    raise ValueError(f"Unknown transcription backend: {config.backend!r}")
