"""On-device speaker embeddings, enrollment, and roster-scoped matching."""

from __future__ import annotations

import sys
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np

from .attribution import Speaker
from .embeddings import cosine, cosine_scores, normalize
from .speaker_store import (
    DEFAULT_MODEL_KEY,
    MIN_SAMPLE_SPEECH_SECONDS,
    SpeakerProfile,
    SpeakerStore,
    profile_state,
)
from .types import AttributionSource, AudioChunk, Channel, ProfileState

MODEL_FILENAME = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
MIN_ENROLLMENT_SECONDS = MIN_SAMPLE_SPEECH_SECONDS
MIN_ATTRIBUTION_SECONDS = 0.7
MANUAL_ENROLLMENT_SECONDS = 3.0
MAX_CLIPPED_RATIO = 0.01
SOURCE_THRESHOLD = 0.60
GLOBAL_THRESHOLD = 0.65
MATCH_MARGIN = 0.05
NEAREST_THRESHOLD = 0.45
NEAREST_MARGIN = 0.02

SampleRejection = Literal[
    "speaker_missing",
    "too_short",
    "clipped",
    "invalid_embedding",
    "inconsistent",
    "ambiguous_profile",
]


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
        return normalize(self._extractor.compute(stream))


@dataclass(frozen=True)
class AttributionDecision:
    name: str
    speaker_index: int | None
    speaker_id: str | None
    source: AttributionSource
    confidence: float | None = None
    profile: SpeakerProfile | None = None


@dataclass(frozen=True)
class ManualSampleResult:
    accepted: bool
    reason: SampleRejection | None = None
    profile: SpeakerProfile | None = None


@dataclass(frozen=True)
class _TemporarySample:
    embedding: np.ndarray
    source: Channel
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
        if (
            self._engine is None
            or chunk_speech_seconds(chunk) < MIN_ATTRIBUTION_SECONDS
        ):
            return False
        if chunk_quality(chunk) < 1.0 - MAX_CLIPPED_RATIO:
            return False
        if chunk.manual_speaker_index is not None:
            return False
        return len(self._members()) > 1

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
            return self._decision(manual, "manual", 1.0)
        if len(members) == 1:
            return self._decision(members[0], "single", 1.0)
        if embedding is not None:
            match = self._match(embedding, members, chunk.channel)
            if match is not None:
                return match
        return AttributionDecision("Unknown", None, None, "unknown", None)

    def _decision(
        self, index: int, source: AttributionSource, confidence: float
    ) -> AttributionDecision:
        speaker = self._roster[index]
        return AttributionDecision(speaker.name, index, speaker.speaker_id, source, confidence)

    def _match(
        self, embedding: np.ndarray, members: list[int], channel: Channel
    ) -> AttributionDecision | None:
        """The best-scoring enrolled member: a confident ``auto`` match, or a
        ``nearest`` guess when nobody clears the bar but one person is
        clearly closest. None when it is a toss-up or nobody is close."""
        ids = [
            speaker_id
            for index in members
            if (speaker_id := self._roster[index].speaker_id) is not None
        ]
        vectors = self._store.match_vectors(ids, channel)
        scored = ((cosine(embedding, vector.embedding), vector) for vector in vectors)
        scores = sorted(
            ((score, vector) for score, vector in scored if score is not None),
            key=lambda item: item[0],
            reverse=True,
        )
        if not scores:
            return None
        best_score, best = scores[0]
        threshold = SOURCE_THRESHOLD if best.source_specific else GLOBAL_THRESHOLD
        margin = best_score - scores[1][0] if len(scores) > 1 else 1.0
        index = next(
            (item for item in members if self._roster[item].speaker_id == best.speaker_id), None
        )
        if index is None:
            return None
        if best_score >= threshold and margin >= MATCH_MARGIN:
            return self._decision(index, "auto", best_score)
        if best_score >= NEAREST_THRESHOLD and margin >= NEAREST_MARGIN:
            return self._decision(index, "nearest", best_score)
        return None

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

        try:
            vector = normalize(embedding)
        except ValueError:
            return ManualSampleResult(False, "invalid_embedding")

        speaker = self._roster[speaker_index]
        if speaker.speaker_id:
            profile = self._store.profile(speaker.speaker_id)
            if profile.state == "ready":
                own = self._store.profile_vector(speaker.speaker_id, chunk.channel)
                if own is None:
                    return ManualSampleResult(False, "inconsistent")
                own_score = cosine(vector, own.embedding)
                if own_score is None:
                    return ManualSampleResult(False, "inconsistent")
                threshold = SOURCE_THRESHOLD if own.source_specific else GLOBAL_THRESHOLD
                if own_score < threshold:
                    return ManualSampleResult(False, "inconsistent")
                competitor_ids = [
                    item.speaker_id
                    for index, item in enumerate(self._roster)
                    if index != speaker_index and item.speaker_id
                ]
                competitors = self._store.match_vectors(competitor_ids, chunk.channel)
                competitor_scores = cosine_scores(
                    vector, (item.embedding for item in competitors)
                )
                if competitor_scores and own_score - max(competitor_scores) < MATCH_MARGIN:
                    return ManualSampleResult(False, "ambiguous_profile")
            profile = self._store.add_sample(
                speaker.speaker_id, vector, chunk.channel, seconds, quality
            )
            return ManualSampleResult(True, profile=profile)

        self._temporary.setdefault(speaker_index, []).append(
            _TemporarySample(
                np.array(vector, dtype=np.float32, copy=True), chunk.channel, seconds, quality
            )
        )
        return ManualSampleResult(True)

    def reset_profile(self, index: int) -> None:
        """Throw away everything learned about this person's voice.

        One bad first clip drags the centroid far enough that every later
        sample fails the consistency check in ``add_manual_sample`` and is
        rejected — so the profile can only get worse. Starting over is the
        only way back.
        """
        if not 0 <= index < len(self._roster):
            raise KeyError("speaker not found")
        speaker = self._roster[index]
        self._temporary.pop(index, None)
        if speaker.speaker_id:
            self._store.reset_profile(speaker.speaker_id)

    def profile_status(self, index: int) -> tuple[ProfileState, float]:
        speaker = self._roster[index]
        if speaker.speaker_id:
            profile = self._store.profile(speaker.speaker_id)
            return profile.state, profile.enrollment_seconds
        seconds = sum(sample.speech_seconds for sample in self._temporary.get(index, ()))
        return profile_state(seconds), seconds

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
