"""Wires capture → VAD chunking → transcription → sink across threads.

Producers (audio callbacks or a file reader) put channel-tagged, stamped
frame blocks on ``frame_queue``. A chunker thread routes each block to that
channel's VadChunker (each channel keeps its own VAD state); a single
transcription worker turns chunks into TranscriptSegments and hands them to
the sink. distil-small.en INT8 runs faster than realtime on CPU, so one
worker keeps up and the transcript lags live audio by a few seconds at most.

Streams are never mixed: mic and loopback frames stay separate through the
whole pipeline, which is what makes remote-speaker attribution free.

``None`` on a queue is the end-of-stream sentinel; ``finish()`` drains
everything (including a final VAD flush per channel) before returning.
"""

from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass, replace
from typing import Callable

from .attribution import Attributor
from .chunker import Vad, VadChunker
from .clock import SessionClock
from .config import Config
from .transcriber import Transcriber
from .types import AudioChunk, Channel, TranscriptSegment
from .vad import SileroVad

Sink = Callable[[TranscriptSegment, float], None]
"""Receives each segment plus its latency (seconds behind live) at delivery."""


@dataclass(frozen=True)
class _Split:
    """Control message: force a chunk boundary on one channel."""

    channel: Channel


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
    ) -> None:
        self._config = config
        self._clock = clock
        self._transcriber = transcriber
        self._sink = sink
        self._attributor = attributor
        self.frame_queue: queue.Queue = queue.Queue(maxsize=512)
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=64)
        self._chunkers = {
            channel: VadChunker(vad_factory(), config, channel)
            for channel in channels
        }
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

    def _run_chunker(self) -> None:
        while True:
            block = self.frame_queue.get()
            if block is None:
                for chunker in self._chunkers.values():
                    final_chunk = chunker.flush()
                    if final_chunk is not None:
                        self._chunk_queue.put(final_chunk)
                self._chunk_queue.put(None)
                return
            if isinstance(block, _Split):
                forced = self._chunkers[block.channel].split()
                if forced is not None:
                    self._chunk_queue.put(forced)
                continue
            channel, timestamp, samples = block
            for chunk in self._chunkers[channel].push(timestamp, samples):
                self._chunk_queue.put(chunk)

    def _run_worker(self) -> None:
        while True:
            chunk: AudioChunk | None = self._chunk_queue.get()
            if chunk is None:
                return
            try:
                segment = self._transcriber.transcribe(chunk)
            except Exception as error:
                print(f"transcription error: {error}", file=sys.stderr)
                continue
            if self._attributor is not None:
                segment = replace(segment, speaker=self._attributor.for_chunk(chunk))
            if segment.text:
                latency = self._clock.now() - chunk.end
                self._sink(segment, latency)
