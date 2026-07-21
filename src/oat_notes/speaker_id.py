"""On-device speaker embeddings, enrollment, and roster-scoped matching."""

from __future__ import annotations

import sys
import threading
from abc import ABC, abstractmethod
from collections import deque
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
MANUAL_ENROLLMENT_SECONDS = 3.0
MAX_CLIPPED_RATIO = 0.01
SOURCE_THRESHOLD = 0.60
GLOBAL_THRESHOLD = 0.65
MATCH_MARGIN = 0.05


def chunk_speech_seconds(chunk: AudioChunk) -> float:
    return float(
        chunk.speech_seconds if chunk.speech_seconds is not None else chunk.duration
    )


def chunk_quality(chunk: AudioChunk) -> float:
    """1.0 is clean; approaches 0.0 as more of the chunk reads as clipped."""
    if chunk.samples.size == 0:
        return 0.0
    clipped = float(np.mean(np.abs(chunk.samples) >= 0.999))
    return max(0.0, 1.0 - clipped)


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

        if sys.platform == "win32":
            lib_dir = Path(sherpa_onnx.__file__).resolve().parent / "lib"
            required = ("onnxruntime.dll", "sherpa-onnx-cxx-api.dll")
            missing = [name for name in required if not (lib_dir / name).is_file()]
            if missing:
                raise RuntimeError(
                    "sherpa-onnx native runtime is incomplete; reinstall "
                    "sherpa-onnx-core (missing " + ", ".join(missing) + ")"
                )
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
class ManualSampleResult:
    accepted: bool
    reason: str | None = None
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
        tracking_engine: SpeakerEmbeddingEngine | None = None,
    ) -> None:
        self._roster = roster
        self._store = store
        self._engine = engine
        self._sample_rate = sample_rate
        self._temporary: dict[int, list[_TemporarySample]] = {}
        # Each bundled extractor instance owns mutable stream state, so calls
        # against the same instance must be serialized. Turn attribution and
        # rolling speaker tracking get their own instance (when the caller
        # supplies one) so they no longer queue behind each other.
        self._embed_lock = threading.Lock()
        self._tracking_engine = tracking_engine or engine
        self._tracking_embed_lock = (
            threading.Lock() if tracking_engine is not None else self._embed_lock
        )

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    def _members(self) -> list[int]:
        return list(range(len(self._roster)))

    def wants_embedding(self, chunk: AudioChunk) -> bool:
        if self._engine is None or chunk_speech_seconds(chunk) < MIN_ENROLLMENT_SECONDS:
            return False
        if chunk_quality(chunk) < 1.0 - MAX_CLIPPED_RATIO:
            return False
        if chunk.manual_speaker_index is not None:
            # Manual enrollment is handled by the stability-gated learner,
            # independently from transcript attribution.
            return False
        member_ids = [
            self._roster[index].speaker_id
            for index in self._members()
            if self._roster[index].speaker_id
        ]
        return bool(self._store.match_vectors(member_ids, chunk.channel.value))

    def embed(self, chunk: AudioChunk) -> np.ndarray:
        if self._engine is None:
            raise RuntimeError("speaker recognition is unavailable")
        with self._embed_lock:
            return self._engine.embed(chunk.samples, self._sample_rate)

    def embed_for_tracking(self, chunk: AudioChunk) -> np.ndarray:
        if self._tracking_engine is None:
            raise RuntimeError("speaker recognition is unavailable")
        with self._tracking_embed_lock:
            return self._tracking_engine.embed(chunk.samples, self._sample_rate)

    def resolve(
        self, chunk: AudioChunk, embedding: np.ndarray | None
    ) -> AttributionDecision:
        members = self._members()
        manual = chunk.manual_speaker_index
        if manual is not None and manual in members:
            speaker = self._roster[manual]
            return AttributionDecision(
                speaker.name,
                manual,
                speaker.speaker_id,
                "manual",
                1.0,
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

    def add_manual_sample(
        self,
        speaker_index: int,
        chunk: AudioChunk,
        embedding: np.ndarray,
    ) -> ManualSampleResult:
        """Persist one stability-gated sample, rejecting unsafe updates."""
        if not 0 <= speaker_index < len(self._roster):
            return ManualSampleResult(False, "speaker_missing")
        seconds = chunk_speech_seconds(chunk)
        quality = chunk_quality(chunk)
        if seconds < MANUAL_ENROLLMENT_SECONDS:
            return ManualSampleResult(False, "too_short")
        if quality < 1.0 - MAX_CLIPPED_RATIO:
            return ManualSampleResult(False, "clipped")

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            return ManualSampleResult(False, "invalid_embedding")
        vector = np.ascontiguousarray(vector / norm, dtype=np.float32)

        speaker = self._roster[speaker_index]
        if speaker.speaker_id:
            profile = self._store.profile(speaker.speaker_id)
            if profile.state == "ready":
                own = self._store.profile_vector(
                    speaker.speaker_id, chunk.channel.value
                )
                if own is None or own.embedding.shape != vector.shape:
                    return ManualSampleResult(False, "inconsistent")
                own_score = float(np.dot(vector, own.embedding))
                threshold = SOURCE_THRESHOLD if own.source_specific else GLOBAL_THRESHOLD
                if own_score < threshold:
                    return ManualSampleResult(False, "inconsistent")
                competitor_ids = [
                    item.speaker_id
                    for index, item in enumerate(self._roster)
                    if index != speaker_index and item.speaker_id
                ]
                competitors = self._store.match_vectors(
                    competitor_ids, chunk.channel.value
                )
                competitor_scores = [
                    float(np.dot(vector, item.embedding))
                    for item in competitors
                    if item.embedding.shape == vector.shape
                ]
                if competitor_scores and own_score - max(competitor_scores) < MATCH_MARGIN:
                    return ManualSampleResult(False, "ambiguous_profile")
            profile = self._store.add_sample(
                speaker.speaker_id,
                vector,
                chunk.channel.value,
                seconds,
                quality,
            )
            return ManualSampleResult(True, profile=profile)

        self._temporary.setdefault(speaker_index, []).append(
            _TemporarySample(
                np.array(vector, dtype=np.float32, copy=True),
                chunk.channel.value,
                seconds,
                quality,
            )
        )
        return ManualSampleResult(True)

    def profile_status(self, index: int) -> tuple[str, float]:
        speaker = self._roster[index]
        if speaker.speaker_id:
            profile = self._store.profile(speaker.speaker_id)
            return profile.state, profile.enrollment_seconds
        seconds = sum(sample.speech_seconds for sample in self._temporary.get(index, ()))
        state = "untrained" if seconds <= 0 else "ready" if seconds >= 5.0 else "learning"
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


class RollingSpeakerBuffer:
    """Build overlapping, speech-qualified windows without changing ASR chunks."""

    def __init__(
        self,
        channel: Channel,
        sample_rate: int,
        window_seconds: float,
        hop_seconds: float,
        min_speech_seconds: float = MIN_ENROLLMENT_SECONDS,
    ) -> None:
        self._channel = channel
        self._sample_rate = sample_rate
        self._window_samples = max(1, round(window_seconds * sample_rate))
        self._hop_samples = max(1, round(hop_seconds * sample_rate))
        self._min_speech_samples = max(1, round(min_speech_seconds * sample_rate))
        self._windows: deque[tuple[float, np.ndarray, bool]] = deque()
        self._sample_count = 0
        self._samples_since_check = 0
        self._ready = False

    def reset(self) -> None:
        self._windows.clear()
        self._sample_count = 0
        self._samples_since_check = 0
        self._ready = False

    def push(
        self, timestamp: float, samples: np.ndarray, is_speech: bool
    ) -> AudioChunk | None:
        block = np.asarray(samples, dtype=np.float32)
        self._windows.append((timestamp, block, is_speech))
        self._sample_count += block.size
        self._samples_since_check += block.size

        # Keep the shortest whole-block window that still covers the target.
        while (
            len(self._windows) > 1
            and self._sample_count - self._windows[0][1].size
            >= self._window_samples
        ):
            _, removed, _ = self._windows.popleft()
            self._sample_count -= removed.size

        if self._sample_count < self._window_samples:
            return None
        if self._ready and self._samples_since_check < self._hop_samples:
            return None
        self._ready = True
        self._samples_since_check = 0

        speech_samples = sum(
            window.size for _, window, speech in self._windows if speech
        )
        if speech_samples < self._min_speech_samples:
            return None
        start = self._windows[0][0]
        waveform = np.concatenate([window for _, window, _ in self._windows])
        return AudioChunk(
            samples=waveform,
            channel=self._channel,
            start=start,
            end=start + waveform.size / self._sample_rate,
            speech_seconds=speech_samples / self._sample_rate,
        )


class SpeakerChangeGate:
    """Require stable repeated matches before publishing a speaker change."""

    def __init__(self, confirmations: int = 2) -> None:
        self._confirmations = max(1, confirmations)
        self._current: int | None = None
        self._candidate: int | None = None
        self._count = 0

    @property
    def current(self) -> int | None:
        return self._current

    def force(self, speaker_index: int | None) -> None:
        self._current = speaker_index
        self._candidate = None
        self._count = 0

    def observe(self, decision: AttributionDecision) -> AttributionDecision | None:
        index = decision.speaker_index
        if decision.source != "auto" or index is None:
            self._candidate = None
            self._count = 0
            return None
        if index == self._current:
            self._candidate = None
            self._count = 0
            return None
        if index == self._candidate:
            self._count += 1
        else:
            self._candidate = index
            self._count = 1
        if self._count < self._confirmations:
            return None
        self._current = index
        self._candidate = None
        self._count = 0
        return decision
