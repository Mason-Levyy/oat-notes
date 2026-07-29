"""Threads and queues: frame blocks → per-channel VAD chunkers → one
transcription worker → sink. Channels never mix; ``None`` is the
end-of-stream sentinel on every queue."""

from __future__ import annotations

import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .attribution import Attributor
from .chunker import Vad, VadChunker
from .clock import SessionClock
from .config import Config
from .speaker_id import RollingSpeakerBuffer, SpeakerChangeGate, SpeakerResolver
from .transcriber import Transcriber
from .types import AudioChunk, Channel, TranscriptSegment
from .vad import SileroVad

Sink = Callable[[TranscriptSegment, float, "np.ndarray | None"], None]
TrackingSink = Callable[[int, Channel, str, float | None], None]


@dataclass(frozen=True)
class ProfileLearningUpdate:
    phase: str
    speaker_index: int
    source: Channel | None
    speech_seconds: float
    target_seconds: float
    reason: str | None = None
    profile_state: str | None = None
    enrollment_seconds: float | None = None

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "speaker_index": self.speaker_index,
            "source": self.source.value if self.source is not None else None,
            "speech_seconds": round(self.speech_seconds, 2),
            "target_seconds": self.target_seconds,
            "reason": self.reason,
            "profile_state": self.profile_state,
            "enrollment_seconds": (
                round(self.enrollment_seconds, 2)
                if self.enrollment_seconds is not None
                else None
            ),
        }


ProfileLearningSink = Callable[[ProfileLearningUpdate], None]


@dataclass(frozen=True)
class _Split:
    channel: Channel
    speaker_index: int | None = None


@dataclass(frozen=True)
class _BeginProfileLearning:
    speaker_index: int
    selected_at: float


@dataclass(frozen=True)
class _CancelProfileLearning:
    pass


@dataclass
class _ProfileCandidate:
    generation: int
    speaker_index: int
    started_at: float
    source: Channel | None = None
    samples: list[np.ndarray] = field(default_factory=list)
    speech_seconds: float = 0.0
    last_reported_step: int = -1


@dataclass(frozen=True)
class _ProfileJob:
    generation: int
    speaker_index: int
    chunk: AudioChunk


class Pipeline:
    def __init__(
        self,
        config: Config,
        clock: SessionClock,
        transcriber: Transcriber,
        sink: Sink,
        channels: tuple[Channel, ...] = (Channel.MIC,),
        vad_factory: Callable[[], Vad] = SileroVad,
        attributor: Attributor | None = None,
        speaker_resolver: SpeakerResolver | None = None,
        speaker_tracking_sink: TrackingSink | None = None,
        profile_learning_sink: ProfileLearningSink | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._transcriber = transcriber
        self._sink = sink
        self._attributor = attributor
        self._speaker_resolver = speaker_resolver
        self._speaker_tracking_sink = speaker_tracking_sink
        self._profile_learning_sink = profile_learning_sink
        self._channels = channels
        self.frame_queue: queue.Queue = queue.Queue(maxsize=512)
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=64)
        self._tracking_lock = threading.Lock()
        tracking_enabled = bool(
            speaker_resolver is not None
            and speaker_resolver.enabled
            and speaker_tracking_sink is not None
        )
        learning_enabled = bool(
            speaker_resolver is not None
            and speaker_resolver.enabled
            and profile_learning_sink is not None
        )
        self._tracking_enabled = tracking_enabled
        self._learning_enabled = learning_enabled
        self._tracking_queue: queue.Queue | None = (
            queue.Queue(maxsize=8) if tracking_enabled else None
        )
        self._tracking_buffers = (
            {
                channel: RollingSpeakerBuffer(
                    channel,
                    config.sample_rate,
                    config.speaker_window_seconds,
                    config.speaker_hop_seconds,
                    config.speaker_min_speech_seconds,
                )
                for channel in channels
            }
            if tracking_enabled
            else {}
        )
        self._tracking_gates = (
            {
                channel: SpeakerChangeGate(config.speaker_confirmations)
                for channel in channels
            }
            if tracking_enabled
            else {}
        )
        self._tracking_generations = {channel: 0 for channel in channels}
        self._chunkers = {
            channel: VadChunker(
                vad_factory(),
                config,
                channel,
                window_sink=(
                    lambda timestamp, samples, is_speech, channel=channel:
                    self._handle_vad_window(channel, timestamp, samples, is_speech)
                )
                if tracking_enabled or learning_enabled
                else None,
            )
            for channel in channels
        }
        self._pending_manual: dict[Channel, int | None] = {
            channel: None for channel in channels
        }
        self._manual_turns: dict[tuple[Channel, int], int] = {}
        self._learning_lock = threading.Lock()
        self._learning_generation = 0
        self._profile_candidate: _ProfileCandidate | None = None
        self._learning_blocked_channels: set[Channel] = set()
        self._recent_speech: dict[Channel, float | None] = {
            channel: None for channel in channels
        }
        self._profile_queue: queue.Queue | None = (
            queue.Queue(maxsize=4) if learning_enabled else None
        )
        self._chunker_thread = threading.Thread(
            target=self._run_chunker, name="chunker", daemon=True
        )
        self._worker_thread = threading.Thread(
            target=self._run_worker, name="transcriber", daemon=True
        )
        self._tracking_thread = (
            threading.Thread(
                target=self._run_speaker_tracker,
                name="speaker-tracker",
                daemon=True,
            )
            if tracking_enabled
            else None
        )
        self._profile_thread = (
            threading.Thread(
                target=self._run_profile_worker,
                name="speaker-profile-learner",
                daemon=True,
            )
            if learning_enabled
            else None
        )

    def start(self) -> None:
        self._chunker_thread.start()
        self._worker_thread.start()
        if self._tracking_thread is not None:
            self._tracking_thread.start()
        if self._profile_thread is not None:
            self._profile_thread.start()

    def finish(self) -> None:
        """Signal end of input and block until every queued chunk is out."""
        self.frame_queue.put(None)
        self._chunker_thread.join()
        self._worker_thread.join()
        if self._tracking_thread is not None:
            self._tracking_thread.join()
        if self._profile_thread is not None:
            self._profile_thread.join()

    def split_channel(self, channel: Channel) -> None:
        """Cut the in-flight chunk on ``channel`` right now (hotkey press).

        Called from the hotkey listener thread; must never block it.
        """
        try:
            self.frame_queue.put_nowait(_Split(channel))
        except queue.Full:
            pass

    def manual_override(self, channel: Channel, speaker_index: int) -> None:
        """Cut now and force only the next/current VAD turn to one speaker."""
        self._reset_tracking(channel, speaker_index)
        try:
            self.frame_queue.put_nowait(_Split(channel, speaker_index))
        except queue.Full:
            pass

    def begin_profile_learning(self, speaker_index: int, selected_at: float) -> None:
        """Start one non-blocking, stability-gated manual enrollment attempt."""
        if not self._learning_enabled:
            return
        try:
            self.frame_queue.put_nowait(
                _BeginProfileLearning(speaker_index, selected_at)
            )
        except queue.Full:
            self._emit_profile_learning(
                ProfileLearningUpdate(
                    "skipped",
                    speaker_index,
                    None,
                    0.0,
                    self._config.manual_enrollment_seconds,
                    reason="busy",
                )
            )

    def cancel_profile_learning(self) -> None:
        """Cancel the in-flight manual sample capture, if any, right now.

        Routed through ``frame_queue`` — like ``begin_profile_learning`` and
        ``manual_override`` — so it's ordered relative to any audio already
        queued instead of racing the chunker thread from the caller's own
        thread.
        """
        try:
            self.frame_queue.put_nowait(_CancelProfileLearning())
        except queue.Full:
            pass

    def _cancel_profile_learning(self) -> None:
        update = None
        resume = None
        with self._learning_lock:
            candidate = self._profile_candidate
            if candidate is None:
                return
            self._profile_candidate = None
            if candidate.source is not None:
                self._learning_blocked_channels.discard(candidate.source)
                resume = (candidate.source, candidate.speaker_index)
            update = ProfileLearningUpdate(
                "skipped",
                candidate.speaker_index,
                candidate.source,
                candidate.speech_seconds,
                self._config.manual_enrollment_seconds,
                reason="cancelled",
            )
        self._emit_profile_learning(update)
        if resume is not None:
            self._reset_tracking(*resume)

    def reset_speaker_tracking(self, speaker_index: int | None = None) -> None:
        """Drop live tracking state on every channel.

        Called when a voice profile is wiped mid-meeting: the rolling buffers
        and the change gate still hold matches made against a centroid that no
        longer exists. ``None`` genuinely clears the current speaker rather
        than pinning a stale one.
        """
        for channel in self._channels:
            self._reset_tracking(channel, speaker_index)

    def _reset_tracking(self, channel: Channel, speaker_index: int | None) -> None:
        if self._tracking_queue is None:
            return
        with self._tracking_lock:
            self._tracking_generations[channel] += 1
            self._tracking_buffers[channel].reset()
            self._tracking_gates[channel].force(speaker_index)

    def _emit_profile_learning(self, update: ProfileLearningUpdate) -> None:
        if self._profile_learning_sink is not None:
            self._profile_learning_sink(update)

    def _recent_active_channels(self, timestamp: float) -> list[Channel]:
        tolerance = self._config.silence_split_seconds + self._config.window_seconds
        return [
            channel
            for channel, last_speech in self._recent_speech.items()
            if last_speech is not None
            and 0.0 <= timestamp - last_speech <= tolerance
        ]

    def _begin_profile_learning(self, command: _BeginProfileLearning) -> None:
        update = None
        with self._learning_lock:
            self._learning_generation += 1
            generation = self._learning_generation
            self._learning_blocked_channels.clear()
            candidate = _ProfileCandidate(
                generation, command.speaker_index, command.selected_at
            )
            active = self._recent_active_channels(command.selected_at)
            if len(active) > 1:
                self._profile_candidate = None
                update = ProfileLearningUpdate(
                    "skipped",
                    command.speaker_index,
                    None,
                    0.0,
                    self._config.manual_enrollment_seconds,
                    reason="ambiguous_source",
                )
            else:
                if active:
                    candidate.source = active[0]
                    self._learning_blocked_channels.add(active[0])
                self._profile_candidate = candidate
                update = ProfileLearningUpdate(
                    "collecting",
                    command.speaker_index,
                    candidate.source,
                    0.0,
                    self._config.manual_enrollment_seconds,
                )
        self._emit_profile_learning(update)

    def _handle_vad_window(
        self,
        channel: Channel,
        timestamp: float,
        samples: np.ndarray,
        is_speech: bool,
    ) -> None:
        if self._learning_enabled:
            self._observe_profile_window(channel, timestamp, samples, is_speech)
        if self._tracking_enabled:
            self._track_window(channel, timestamp, samples, is_speech)

    def _observe_profile_window(
        self,
        channel: Channel,
        timestamp: float,
        samples: np.ndarray,
        is_speech: bool,
    ) -> None:
        update = None
        job = None
        resume = None
        with self._learning_lock:
            if is_speech:
                self._recent_speech[channel] = timestamp
            candidate = self._profile_candidate
            if candidate is None:
                return
            if (
                timestamp - candidate.started_at
                >= self._config.manual_enrollment_timeout_seconds
            ):
                self._profile_candidate = None
                if candidate.source is not None:
                    self._learning_blocked_channels.discard(candidate.source)
                    resume = (candidate.source, candidate.speaker_index)
                update = ProfileLearningUpdate(
                    "skipped",
                    candidate.speaker_index,
                    candidate.source,
                    candidate.speech_seconds,
                    self._config.manual_enrollment_seconds,
                    reason="timeout",
                )
            elif not is_speech:
                return
            else:
                if candidate.source is None:
                    active = self._recent_active_channels(timestamp)
                    if len(active) > 1:
                        self._profile_candidate = None
                        update = ProfileLearningUpdate(
                            "skipped",
                            candidate.speaker_index,
                            None,
                            0.0,
                            self._config.manual_enrollment_seconds,
                            reason="ambiguous_source",
                        )
                    else:
                        candidate.source = channel
                        self._learning_blocked_channels.add(channel)
                        update = ProfileLearningUpdate(
                            "collecting",
                            candidate.speaker_index,
                            channel,
                            0.0,
                            self._config.manual_enrollment_seconds,
                        )
                if self._profile_candidate is candidate and candidate.source == channel:
                    window = np.array(samples, dtype=np.float32, copy=True)
                    candidate.samples.append(window)
                    candidate.speech_seconds += window.size / self._config.sample_rate
                    step = int(candidate.speech_seconds / 0.25)
                    if step > candidate.last_reported_step:
                        candidate.last_reported_step = step
                        update = ProfileLearningUpdate(
                            "collecting",
                            candidate.speaker_index,
                            channel,
                            min(
                                candidate.speech_seconds,
                                self._config.manual_enrollment_seconds,
                            ),
                            self._config.manual_enrollment_seconds,
                        )
                    if (
                        candidate.speech_seconds
                        >= self._config.manual_enrollment_seconds
                    ):
                        target_samples = int(
                            self._config.manual_enrollment_seconds
                            * self._config.sample_rate
                        )
                        waveform = np.concatenate(candidate.samples)[:target_samples]
                        actual_seconds = waveform.size / self._config.sample_rate
                        chunk = AudioChunk(
                            waveform,
                            channel,
                            candidate.started_at,
                            candidate.started_at + actual_seconds,
                            manual_speaker_index=candidate.speaker_index,
                            speech_seconds=actual_seconds,
                        )
                        job = _ProfileJob(
                            candidate.generation,
                            candidate.speaker_index,
                            chunk,
                        )
                        self._profile_candidate = None
        if update is not None:
            self._emit_profile_learning(update)
        if resume is not None:
            self._reset_tracking(*resume)
        if job is not None and self._profile_queue is not None:
            try:
                self._profile_queue.put_nowait(job)
            except queue.Full:
                with self._learning_lock:
                    if job.generation == self._learning_generation:
                        self._learning_blocked_channels.discard(job.chunk.channel)
                self._reset_tracking(job.chunk.channel, job.speaker_index)
                self._emit_profile_learning(
                    ProfileLearningUpdate(
                        "skipped",
                        job.speaker_index,
                        job.chunk.channel,
                        job.chunk.speech_seconds or 0.0,
                        self._config.manual_enrollment_seconds,
                        reason="busy",
                    )
                )

    def _tracking_paused(self, channel: Channel) -> bool:
        with self._learning_lock:
            candidate = self._profile_candidate
            if candidate is not None and (
                candidate.source is None or candidate.source == channel
            ):
                return True
            return channel in self._learning_blocked_channels

    def _track_window(
        self,
        channel: Channel,
        timestamp: float,
        samples,
        is_speech: bool,
    ) -> None:
        tracking_queue = self._tracking_queue
        if tracking_queue is None:
            return
        with self._tracking_lock:
            chunk = self._tracking_buffers[channel].push(
                timestamp, samples, is_speech
            )
            generation = self._tracking_generations[channel]
        if chunk is None:
            return
        item = (generation, chunk)
        try:
            tracking_queue.put_nowait(item)
        except queue.Full:
            # Tracking is live state, so a fresh window is more useful than a
            # stale backlog when inference briefly falls behind.
            try:
                tracking_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                tracking_queue.put_nowait(item)
            except queue.Full:
                pass

    def _queue_chunk(self, chunk: AudioChunk) -> None:
        key = (chunk.channel, chunk.turn_id)
        if key not in self._manual_turns:
            pending = self._pending_manual.get(chunk.channel)
            if pending is not None:
                self._manual_turns[key] = pending
                self._pending_manual[chunk.channel] = None
        manual = self._manual_turns.get(key)
        self._chunk_queue.put(replace(chunk, manual_speaker_index=manual))
        if chunk.turn_end:
            self._manual_turns.pop(key, None)

    def _run_chunker(self) -> None:
        while True:
            block = self.frame_queue.get()
            if block is None:
                with self._learning_lock:
                    candidate = self._profile_candidate
                    self._profile_candidate = None
                    self._learning_blocked_channels.clear()
                if candidate is not None:
                    self._emit_profile_learning(
                        ProfileLearningUpdate(
                            "skipped",
                            candidate.speaker_index,
                            candidate.source,
                            candidate.speech_seconds,
                            self._config.manual_enrollment_seconds,
                            reason="meeting_ended",
                        )
                    )
                for chunker in self._chunkers.values():
                    final_chunk = chunker.flush()
                    if final_chunk is not None:
                        self._queue_chunk(final_chunk)
                self._chunk_queue.put(None)
                if self._tracking_queue is not None:
                    self._tracking_queue.put(None)
                if self._profile_queue is not None:
                    self._profile_queue.put(None)
                return
            if isinstance(block, _BeginProfileLearning):
                self._begin_profile_learning(block)
                continue
            if isinstance(block, _CancelProfileLearning):
                self._cancel_profile_learning()
                continue
            if isinstance(block, _Split):
                forced = self._chunkers[block.channel].split()
                if forced is not None:
                    self._queue_chunk(forced)
                if block.speaker_index is not None:
                    self._pending_manual[block.channel] = block.speaker_index
                continue
            channel, timestamp, samples = block
            for chunk in self._chunkers[channel].push(timestamp, samples):
                self._queue_chunk(chunk)

    def _run_speaker_tracker(self) -> None:
        tracking_queue = self._tracking_queue
        resolver = self._speaker_resolver
        sink = self._speaker_tracking_sink
        if tracking_queue is None or resolver is None or sink is None:
            return
        while True:
            item = tracking_queue.get()
            if item is None:
                return
            generation, chunk = item
            if self._tracking_paused(chunk.channel):
                continue
            with self._tracking_lock:
                if generation != self._tracking_generations[chunk.channel]:
                    continue
            if not resolver.wants_embedding(chunk):
                continue
            try:
                embedding = resolver.embed_for_tracking(chunk)
                decision = resolver.resolve(chunk, embedding)
            except Exception as error:
                print(
                    f"speaker tracking error: {type(error).__name__}",
                    file=sys.stderr,
                )
                continue
            if self._tracking_paused(chunk.channel):
                continue
            with self._tracking_lock:
                if generation != self._tracking_generations[chunk.channel]:
                    continue
                gate = self._tracking_gates[chunk.channel]
                confirmed = gate.observe(decision)
            if confirmed is not None and confirmed.speaker_index is not None:
                self.split_channel(chunk.channel)
                sink(
                    confirmed.speaker_index,
                    chunk.channel,
                    confirmed.source,
                    confirmed.confidence,
                )

    def _run_profile_worker(self) -> None:
        profile_queue = self._profile_queue
        resolver = self._speaker_resolver
        if profile_queue is None or resolver is None:
            return
        while True:
            job = profile_queue.get()
            if job is None:
                return
            with self._learning_lock:
                if job.generation != self._learning_generation:
                    continue
            try:
                embedding = resolver.embed_for_tracking(job.chunk)
            except Exception as error:
                print(
                    f"speaker profile learning error: {type(error).__name__}",
                    file=sys.stderr,
                )
                with self._learning_lock:
                    if job.generation != self._learning_generation:
                        continue
                    self._learning_blocked_channels.discard(job.chunk.channel)
                self._reset_tracking(job.chunk.channel, job.speaker_index)
                self._emit_profile_learning(
                    ProfileLearningUpdate(
                        "skipped",
                        job.speaker_index,
                        job.chunk.channel,
                        job.chunk.speech_seconds or 0.0,
                        self._config.manual_enrollment_seconds,
                        reason="embedding_error",
                    )
                )
                continue

            with self._learning_lock:
                if job.generation != self._learning_generation:
                    continue
                result = resolver.add_manual_sample(
                    job.speaker_index, job.chunk, embedding
                )
                self._learning_blocked_channels.discard(job.chunk.channel)
            state, total_seconds = resolver.profile_status(job.speaker_index)
            self._reset_tracking(job.chunk.channel, job.speaker_index)
            self._emit_profile_learning(
                ProfileLearningUpdate(
                    "saved" if result.accepted else "skipped",
                    job.speaker_index,
                    job.chunk.channel,
                    job.chunk.speech_seconds or 0.0,
                    self._config.manual_enrollment_seconds,
                    reason=result.reason,
                    profile_state=state,
                    enrollment_seconds=total_seconds,
                )
            )

    def _run_worker(self) -> None:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="speaker-id") as pool:
            while True:
                chunk: AudioChunk | None = self._chunk_queue.get()
                if chunk is None:
                    return
                future = None
                if (
                    self._speaker_resolver is not None
                    and self._speaker_resolver.wants_embedding(chunk)
                ):
                    future = pool.submit(self._speaker_resolver.embed, chunk)
                try:
                    segment = self._transcriber.transcribe(chunk)
                except Exception as error:
                    # Exception messages are deliberately omitted: a backend
                    # must not be able to echo captured audio/transcript data
                    # into the installed application's durable log.
                    print(
                        f"transcription error: {type(error).__name__}",
                        file=sys.stderr,
                    )
                    continue
                segment = replace(segment, turn_end=chunk.turn_end)

                embedding = None
                if self._speaker_resolver is not None:
                    if future is not None:
                        try:
                            embedding = future.result()
                        except Exception as error:
                            print(
                                f"speaker recognition error: {type(error).__name__}",
                                file=sys.stderr,
                            )
                    decision = self._speaker_resolver.resolve(chunk, embedding)
                    profile = decision.profile
                    segment = replace(
                        segment,
                        speaker=decision.name,
                        speaker_id=decision.speaker_id,
                        speaker_index=decision.speaker_index,
                        attribution=decision.source,
                        confidence=decision.confidence,
                        profile_state=profile.state if profile else None,
                        enrollment_seconds=(
                            profile.enrollment_seconds if profile else None
                        ),
                    )
                elif self._attributor is not None:
                    segment = replace(
                        segment, speaker=self._attributor.for_chunk(chunk)
                    )
                # A turn-end decision still matters to active-speaker state
                # even when transcription produced no printable text.
                if segment.text or (
                    self._speaker_resolver is not None and segment.turn_end
                ):
                    latency = self._clock.now() - chunk.end
                    self._sink(segment, latency, embedding)
