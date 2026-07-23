"""Transcription backends behind ``create_transcriber``: faster-whisper on
CPU, or OpenVINO GenAI offloading to the Intel NPU/GPU."""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from .config import Config
from .types import AudioChunk, TranscriptSegment


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, chunk: AudioChunk) -> TranscriptSegment: ...


def resolve_openvino_model(source: str, offline: bool) -> str:
    """Local directory or HF repo id; repos land as plain files under
    %LOCALAPPDATA% because the HF cache's symlinks need Developer Mode."""
    if Path(source).is_dir():
        return source
    target = (
        Path(os.environ["LOCALAPPDATA"])
        / "oat-notes"
        / "models"
        / source.replace("/", "--")
    )
    if (target / "config.json").exists():
        return str(target)
    if offline:
        raise RuntimeError(
            f"model {source!r} is not cached locally and --offline forbids"
            " downloading it"
        )
    print(f"downloading {source}…", file=sys.stderr)
    from huggingface_hub import snapshot_download

    return snapshot_download(source, local_dir=target)


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, config: Config) -> None:
        from faster_whisper import WhisperModel

        self._config = config
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
        cache_dir = (
            Path(os.environ["LOCALAPPDATA"])
            / "oat-notes"
            / "openvino-cache"
        )
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
