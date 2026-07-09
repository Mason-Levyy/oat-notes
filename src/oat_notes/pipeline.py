"""Wires capture → VAD chunking → transcription → sink across threads.

Producer (audio callback or file reader) puts stamped frame blocks on
``frame_queue``. A chunker thread turns them into AudioChunks; a single
transcription worker turns chunks into TranscriptSegments and hands them to
the sink. distil-small.en INT8 runs faster than realtime on CPU, so one
worker keeps up and the transcript lags live audio by a few seconds at most.

``None`` on a queue is the end-of-stream sentinel; ``finish()`` drains
everything (including a final VAD flush) before returning.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Callable

from .chunker import VadChunker
from .clock import SessionClock
from .config import Config
from .transcriber import Transcriber
from .types import AudioChunk, Channel, TranscriptSegment
from .vad import SileroVad

Sink = Callable[[TranscriptSegment, float], None]
"""Receives each segment plus its latency (seconds behind live) at delivery."""


class Pipeline:
    def __init__(
        self,
        config: Config,
        clock: SessionClock,
        transcriber: Transcriber,
        sink: Sink,
        channel: Channel = Channel.MIC,
    ) -> None:
        self._config = config
        self._clock = clock
        self._transcriber = transcriber
        self._sink = sink
        self.frame_queue: queue.Queue = queue.Queue(maxsize=256)
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=64)
        self._chunker = VadChunker(SileroVad(), config, channel)
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

    def _run_chunker(self) -> None:
        while True:
            block = self.frame_queue.get()
            if block is None:
                final_chunk = self._chunker.flush()
                if final_chunk is not None:
                    self._chunk_queue.put(final_chunk)
                self._chunk_queue.put(None)
                return
            timestamp, samples = block
            for chunk in self._chunker.push(timestamp, samples):
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
            if segment.text:
                latency = self._clock.now() - chunk.end
                self._sink(segment, latency)
