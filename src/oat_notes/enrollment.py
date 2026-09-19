"""Standalone mic-only voice enrollment: capture, VAD-chunk, embed, and
store voice samples for one saved speaker without running a meeting."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .capture import AudioCapture
from .chunker import Vad, VadChunker
from .clock import SessionClock
from .config import Config
from .speaker_id import (
    MAX_CLIPPED_RATIO,
    MIN_ENROLLMENT_SECONDS,
    SpeakerEmbeddingEngine,
    chunk_quality,
    chunk_speech_seconds,
)
from .speaker_store import SpeakerProfile, SpeakerStore
from .types import AudioChunk, Channel
from .vad import SileroVad

TARGET_ENROLLMENT_SECONDS = 8.0

OnProgress = Callable[["EnrollmentProgress"], None]
EnrollmentPhase = Literal["listening", "captured", "rejected", "done", "stopped", "error"]


@dataclass(frozen=True)
class EnrollmentProgress:
    phase: EnrollmentPhase
    captured_seconds: float = 0.0
    target_seconds: float = TARGET_ENROLLMENT_SECONDS
    reason: str | None = None
    profile: SpeakerProfile | None = None


class VoiceEnrollmentRecorder:
    """Records one saved speaker's mic voice and enrolls it directly,
    independent of any meeting session, transcription, or dual-channel
    capture. ``on_progress`` fires on the recorder's own background
    thread — callers publishing to UI state should keep it quick."""

    def __init__(
        self,
        store: SpeakerStore,
        engine: SpeakerEmbeddingEngine,
        speaker_id: str,
        config: Config,
        on_progress: OnProgress,
        device_index: int | None = None,
        target_seconds: float = TARGET_ENROLLMENT_SECONDS,
        vad_factory: Callable[[], Vad] = SileroVad,
    ) -> None:
        self._store = store
        self._engine = engine
        self._speaker_id = speaker_id
        self._config = config
        self._on_progress = on_progress
        self._device_index = device_index
        self._target_seconds = target_seconds
        self._captured_seconds = 0.0
        self._latest_profile: SpeakerProfile | None = None
        self._done = threading.Event()

        self.frame_queue: queue.Queue = queue.Queue(maxsize=512)
        self._chunker = VadChunker(vad_factory(), config, Channel.MIC)
        self._pa = None
        self._capture: AudioCapture | None = None
        self._thread = threading.Thread(
            target=self._run, name="voice-enrollment", daemon=True
        )

    def start(self) -> None:
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        self._capture = AudioCapture(
            self._pa,
            self._device_index,
            Channel.MIC,
            SessionClock(),
            self._config,
            self.frame_queue,
        )
        self._capture.start()
        self._thread.start()

    def stop(self) -> None:
        """Safe to call any time, including after natural completion —
        whatever was already captured stays saved either way."""
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        self._on_progress(
            EnrollmentProgress(phase="listening", target_seconds=self._target_seconds)
        )
        reached = False
        while not reached:
            item = self.frame_queue.get()
            if item is None:
                break
            _channel, timestamp, samples = item
            for chunk in self._chunker.push(timestamp, samples):
                if self._handle_chunk(chunk):
                    reached = True
                    break
        if not reached:
            final_chunk = self._chunker.flush()
            if final_chunk is not None:
                self._handle_chunk(final_chunk)
        self._teardown()

    def _handle_chunk(self, chunk: AudioChunk) -> bool:
        """Embeds and stores an accepted chunk; returns True at target."""
        seconds = chunk_speech_seconds(chunk)
        quality = chunk_quality(chunk)
        if seconds < MIN_ENROLLMENT_SECONDS or quality < 1.0 - MAX_CLIPPED_RATIO:
            self._on_progress(
                EnrollmentProgress(
                    phase="rejected",
                    captured_seconds=self._captured_seconds,
                    target_seconds=self._target_seconds,
                    reason="too short" if seconds < MIN_ENROLLMENT_SECONDS else "too loud",
                )
            )
            return False
        try:
            embedding = self._engine.embed(chunk.samples, self._config.sample_rate)
            self._latest_profile = self._store.add_sample(
                self._speaker_id, embedding, "mic", seconds, quality
            )
        except Exception as error:
            self._on_progress(
                EnrollmentProgress(phase="error", reason=type(error).__name__)
            )
            return False
        self._captured_seconds += seconds
        self._on_progress(
            EnrollmentProgress(
                phase="captured",
                captured_seconds=self._captured_seconds,
                target_seconds=self._target_seconds,
            )
        )
        return self._captured_seconds >= self._target_seconds

    def _teardown(self) -> None:
        if self._done.is_set():
            return
        self._done.set()
        if self._capture is not None:
            self._capture.stop()
        if self._pa is not None:
            self._pa.terminate()
        if self._captured_seconds > 0:
            self._on_progress(
                EnrollmentProgress(
                    phase="done",
                    captured_seconds=self._captured_seconds,
                    target_seconds=self._target_seconds,
                    profile=self._latest_profile,
                )
            )
        else:
            self._on_progress(
                EnrollmentProgress(
                    phase="stopped",
                    captured_seconds=0.0,
                    target_seconds=self._target_seconds,
                )
            )
