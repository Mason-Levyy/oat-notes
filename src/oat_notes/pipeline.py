"""Threads and queues: frame blocks → per-channel VAD chunkers → one
transcription worker → sink. Channels never mix; ``None`` is the
end-of-stream sentinel on every queue."""

from __future__ import annotations

import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable

from .attribution import Attributor
from .chunker import Vad, VadChunker
from .clock import SessionClock
from .config import Config
from .speaker_id import SpeakerResolver
from .transcriber import Transcriber
from .types import AudioChunk, Channel, TranscriptSegment
from .vad import SileroVad

Sink = Callable[[TranscriptSegment, float], None]


@dataclass(frozen=True)
class _Split:
    channel: Channel
    speaker_index: int | None = None


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
    ) -> None:
        self._config = config
        self._clock = clock
        self._transcriber = transcriber
        self._sink = sink
        self._attributor = attributor
        self._speaker_resolver = speaker_resolver
        self.frame_queue: queue.Queue = queue.Queue(maxsize=512)
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=64)
        self._chunkers = {
            channel: VadChunker(vad_factory(), config, channel)
            for channel in channels
        }
        self._pending_manual: dict[Channel, int | None] = {
            channel: None for channel in channels
        }
        self._manual_turns: dict[tuple[Channel, int], int] = {}
        self._chunker_thread = threading.Thread(
            target=self._run_chunker, name="chunker", daemon=True
        )
        self._worker_thread = threading.Thread(
            target=self._run_worker, name="transcriber", daemon=True
        )

    def start(self) -> None:
        self._chunker_thread.start()
        self._worker_thread.start()

    def finish(self) -> None:
        """Signal end of input and block until every queued chunk is out."""
        self.frame_queue.put(None)
        self._chunker_thread.join()
        self._worker_thread.join()

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
        try:
            self.frame_queue.put_nowait(_Split(channel, speaker_index))
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
                for chunker in self._chunkers.values():
                    final_chunk = chunker.flush()
                    if final_chunk is not None:
                        self._queue_chunk(final_chunk)
                self._chunk_queue.put(None)
                return
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

                if self._speaker_resolver is not None:
                    embedding = None
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
                    self._sink(segment, latency)
