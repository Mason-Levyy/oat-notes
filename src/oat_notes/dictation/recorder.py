"""One dictation utterance: mic -> VAD chunks -> transcriber.

Chunks are transcribed while the user is still talking, so releasing the
hotkey leaves only the tail chunk outstanding.

This deliberately does not reuse ``Pipeline``: dual-channel routing, speaker
identification, attribution and profile learning are all dead weight for a
single-speaker utterance. Everything underneath is shared unchanged.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from ..capture import AudioCapture, FrameBlock
from ..chunker import VadChunker
from ..clock import SessionClock
from ..config import Config
from ..log import error_kind
from ..transcriber import Transcriber, TranscriptionHints
from ..types import AudioChunk, Channel
from .stitch import Piece, stitch_pieces

log = logging.getLogger(__name__)

_JOIN_TIMEOUT_SECONDS = 30.0


class UtteranceRecorder:
    """Reusable across utterances. PyAudio and the Silero ONNX session are
    opened once and kept, since re-creating them would add tens of
    milliseconds to every hotkey press. Only the stream is per-utterance."""

    def __init__(
        self,
        config: Config,
        transcriber: Transcriber,
        device_index: int | None = None,
        level_sink: Callable[[float], None] | None = None,
        vad=None,
        hotwords: str | None = None,
    ) -> None:
        self._config = replace(
            config,
            max_chunk_seconds=config.dictation_max_chunk_seconds,
            silence_split_seconds=config.dictation_silence_split_seconds,
        )
        self._transcriber = transcriber
        self._device_index = device_index
        self._level_sink = level_sink
        self.hotwords = hotwords or None

        self._pa = None
        self._vad = vad
        self._chunker: VadChunker | None = None
        self._capture: AudioCapture | None = None
        self._thread: threading.Thread | None = None
        self.frame_queue: queue.Queue[FrameBlock | None] = queue.Queue(maxsize=512)
        self._pieces: list[Piece] = []
        self._speech_resumes_at: float | None = None
        self._speech_seconds = 0.0
        self._cancelled = False

    def prepare(self) -> None:
        """Open the handles that survive between utterances."""
        if self._pa is None:
            import pyaudiowpatch as pyaudio

            self._pa = pyaudio.PyAudio()
        if self._vad is None:
            from ..vad import SileroVad

            self._vad = SileroVad()

    def reopen_audio_host(self) -> None:
        """PortAudio enumerates devices only when its first handle opens, so a
        handle held since launch pins every later one to the launch-time
        device list. Closing and reopening it while idle rescans."""
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        self.prepare()

    def close(self) -> None:
        self.cancel()
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        self._vad = None

    @property
    def speech_seconds(self) -> float:
        """Voiced audio seen so far, used to tell a deliberate quick tap from
        a hold that captured actual speech."""
        return self._speech_seconds

    def start(self) -> None:
        """Open the microphone and start transcribing."""
        self.prepare()
        self.start_worker()
        self._capture = AudioCapture(
            self._pa,
            self._device_index,
            Channel.MIC,
            SessionClock(),
            self._config,
            self.frame_queue,
        )
        self._capture.start()

    def start_worker(self) -> None:
        """Consume ``frame_queue`` without opening a stream. ``start`` builds
        on this; tests push synthetic audio through it directly."""
        self._reset()
        self._chunker = VadChunker(
            self._vad, self._config, Channel.MIC, window_sink=self._note_window
        )
        self._thread = threading.Thread(
            target=self._run, name="dictation-transcribe", daemon=True
        )
        self._thread.start()

    def stop(self) -> str:
        """Flush the tail chunk, drain the worker, and return the utterance."""
        self._teardown()
        return stitch_pieces(
            self._pieces,
            self._config.dictation_clause_gap_seconds,
            self._config.dictation_sentence_gap_seconds,
        )

    def cancel(self) -> None:
        self._cancelled = True
        self._teardown()
        self._pieces.clear()

    def _reset(self) -> None:
        self._cancelled = False
        self._pieces = []
        self._speech_resumes_at = None
        self._speech_seconds = 0.0
        self.frame_queue = queue.Queue(maxsize=512)
        if self._vad is not None:
            self._vad.reset()

    def _teardown(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        if self._thread is not None:
            self.frame_queue.put(None)
            self._thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                log.warning("dictation worker did not finish in time")
            self._thread = None

    def _run(self) -> None:
        while True:
            item = self.frame_queue.get()
            if item is None:
                break
            _, timestamp, samples = item
            if self._level_sink is not None:
                self._report_level(samples)
            for chunk in self._chunker.push(timestamp, samples):
                self._transcribe(chunk)
        if self._cancelled:
            return
        tail = self._chunker.flush()
        if tail is not None:
            self._transcribe(tail)

    def _transcribe(self, chunk: AudioChunk) -> None:
        if self._cancelled:
            return
        try:
            segment = self._transcriber.transcribe(chunk, self._hints())
        except Exception as error:
            log.error("dictation transcription failed: %s", error_kind(error))
            return
        text = segment.text.strip()
        if text:
            self._pieces.append(Piece(text, self._pause_before(chunk)))
        self._speech_resumes_at = chunk.end if chunk.turn_end else None

    def _pause_before(self, chunk: AudioChunk) -> float:
        """Silence between the last speech and this chunk's first speech. A
        chunk closed by silence carries ``silence_split_seconds`` of quiet at
        its tail, and the next opens with pre-roll, so both are given back."""
        if self._speech_resumes_at is None:
            return 0.0
        boundary_quiet = (
            self._config.silence_split_seconds
            + self._config.pre_roll_windows * self._config.window_seconds
        )
        return max(0.0, chunk.start - self._speech_resumes_at + boundary_quiet)

    def _hints(self) -> TranscriptionHints:
        """What was just said, so Whisper continues the sentence rather than
        opening a new one, plus the vocabulary spellings for every chunk."""
        spoken = " ".join(piece.text for piece in self._pieces)
        prompt = spoken[-self._config.dictation_prompt_chars :] or None
        return TranscriptionHints(prompt=prompt, hotwords=self.hotwords)

    def _note_window(self, window_time: float, window, is_speech: bool) -> None:
        if is_speech:
            self._speech_seconds += self._config.window_seconds

    def _report_level(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        level = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        try:
            self._level_sink(level)
        except Exception as error:
            log.warning("level meter disabled: %s", error_kind(error))
            self._level_sink = None
