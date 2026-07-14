"""On-device speaker embeddings, enrollment, and roster-scoped matching."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .attribution import Speaker
from .speaker_store import (
    DEFAULT_MODEL_KEY,
    MIN_SAMPLE_SPEECH_SECONDS,
    SpeakerProfile,
    SpeakerStore,
)
from .types import AudioChunk, Channel

MODEL_FILENAME = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
MIN_ENROLLMENT_SECONDS = MIN_SAMPLE_SPEECH_SECONDS
MAX_CLIPPED_RATIO = 0.01
SOURCE_THRESHOLD = 0.60
GLOBAL_THRESHOLD = 0.65
MATCH_MARGIN = 0.05


class SpeakerEmbeddingEngine(ABC):
    model_key: str = DEFAULT_MODEL_KEY

    @abstractmethod
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


def bundled_model_path() -> Path:
    """Resolve the bundled ONNX asset in source and PyInstaller builds."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "oat_notes" / "assets" / MODEL_FILENAME
    return Path(resources.files("oat_notes") / "assets" / MODEL_FILENAME)


class SherpaOnnxEmbeddingEngine(SpeakerEmbeddingEngine):
    def __init__(self, model_path: Path | None = None) -> None:
        import sherpa_onnx

        path = model_path or bundled_model_path()
        if not path.is_file():
            raise RuntimeError(f"bundled speaker model is missing: {path.name}")
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(path), num_threads=1, debug=False, provider="cpu"
        )
        if not config.validate():
            raise RuntimeError("invalid bundled speaker embedding model")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        waveform = np.ascontiguousarray(samples, dtype=np.float32)
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=waveform)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise ValueError("not enough speech for a speaker embedding")
        vector = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("speaker embedding was empty")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)


@dataclass(frozen=True)
class AttributionDecision:
    name: str
    speaker_index: int | None
    speaker_id: str | None
    source: str
    confidence: float | None = None
    profile: SpeakerProfile | None = None


@dataclass(frozen=True)
class _TemporarySample:
    embedding: np.ndarray
    source: str
    speech_seconds: float
    quality: float


class SpeakerResolver:
    """Resolve one finalized turn without sending captured data anywhere."""

    def __init__(
        self,
        roster: list[Speaker],
        store: SpeakerStore,
        engine: SpeakerEmbeddingEngine | None,
        sample_rate: int,
    ) -> None:
        self._roster = roster
        self._store = store
        self._engine = engine
        self._sample_rate = sample_rate
        self._temporary: dict[int, list[_TemporarySample]] = {}

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    def _members(self) -> list[int]:
        return list(range(len(self._roster)))

    @staticmethod
    def _speech_seconds(chunk: AudioChunk) -> float:
        return float(
            chunk.speech_seconds
            if chunk.speech_seconds is not None
            else chunk.duration
        )

    @staticmethod
    def _quality(chunk: AudioChunk) -> float:
        if chunk.samples.size == 0:
            return 0.0
        clipped = float(np.mean(np.abs(chunk.samples) >= 0.999))
        return max(0.0, 1.0 - clipped)

    def wants_embedding(self, chunk: AudioChunk) -> bool:
        if self._engine is None or self._speech_seconds(chunk) < MIN_ENROLLMENT_SECONDS:
            return False
        if self._quality(chunk) < 1.0 - MAX_CLIPPED_RATIO:
            return False
        if chunk.manual_speaker_index is not None:
            return 0 <= chunk.manual_speaker_index < len(self._roster)
        member_ids = [
            self._roster[index].speaker_id
            for index in self._members()
            if self._roster[index].speaker_id
        ]
        return bool(self._store.match_vectors(member_ids, chunk.channel.value))

    def embed(self, chunk: AudioChunk) -> np.ndarray:
        if self._engine is None:
            raise RuntimeError("speaker recognition is unavailable")
        return self._engine.embed(chunk.samples, self._sample_rate)

    def resolve(
        self, chunk: AudioChunk, embedding: np.ndarray | None
    ) -> AttributionDecision:
        members = self._members()
        manual = chunk.manual_speaker_index
        if manual is not None and manual in members:
            speaker = self._roster[manual]
            profile = None
            if embedding is not None:
                seconds = self._speech_seconds(chunk)
                quality = self._quality(chunk)
                if speaker.speaker_id:
                    profile = self._store.add_sample(
                        speaker.speaker_id,
                        embedding,
                        chunk.channel.value,
                        seconds,
                        quality,
                    )
                else:
                    self._temporary.setdefault(manual, []).append(
                        _TemporarySample(
                            np.array(embedding, dtype=np.float32, copy=True),
                            chunk.channel.value,
                            seconds,
                            quality,
                        )
                    )
            return AttributionDecision(
                speaker.name,
                manual,
                speaker.speaker_id,
                "manual",
                1.0,
                profile,
            )

        if len(members) == 1:
            index = members[0]
            speaker = self._roster[index]
            return AttributionDecision(
                speaker.name, index, speaker.speaker_id, "single", 1.0
            )

        if embedding is not None:
            ids = [
                self._roster[index].speaker_id
                for index in members
                if self._roster[index].speaker_id
            ]
            vectors = self._store.match_vectors(ids, chunk.channel.value)
            scores = sorted(
                (
                    (float(np.dot(embedding, vector.embedding)), vector)
                    for vector in vectors
                    if vector.embedding.shape == embedding.shape
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if scores:
                best_score, best = scores[0]
                threshold = SOURCE_THRESHOLD if best.source_specific else GLOBAL_THRESHOLD
                margin = best_score - scores[1][0] if len(scores) > 1 else 1.0
                if best_score >= threshold and margin >= MATCH_MARGIN:
                    index = next(
                        (
                            item
                            for item in members
                            if self._roster[item].speaker_id == best.speaker_id
                        ),
                        None,
                    )
                    if index is not None:
                        speaker = self._roster[index]
                        return AttributionDecision(
                            speaker.name,
                            index,
                            speaker.speaker_id,
                            "auto",
                            best_score,
                        )

        return AttributionDecision("Unknown", None, None, "unknown", None)

    def profile_status(self, index: int) -> tuple[str, float]:
        speaker = self._roster[index]
        if speaker.speaker_id:
            profile = self._store.profile(speaker.speaker_id)
            return profile.state, profile.enrollment_seconds
        seconds = sum(sample.speech_seconds for sample in self._temporary.get(index, ()))
        state = "untrained" if seconds <= 0 else "ready" if seconds >= 4.0 else "learning"
        return state, seconds

    def persist_guest(self, guest_index: int, speaker_id: str) -> SpeakerProfile:
        samples = self._temporary.pop(guest_index, [])
        profile = self._store.profile(speaker_id)
        for sample in samples:
            profile = self._store.add_sample(
                speaker_id,
                sample.embedding,
                sample.source,
                sample.speech_seconds,
                sample.quality,
            )
        return profile
