"""VAD-driven chunking state machine, pure logic with an injected VAD.

A chunk opens on speech (prepended with pre-roll so word onsets aren't
clipped), closes after ``silence_split_seconds`` of quiet or force-splits at
``max_chunk_seconds``; sub-``min_speech_seconds`` blips are dropped. Window
times re-anchor to each block's capture-time stamp whenever the buffer
drains, so they follow the session clock rather than sample counts.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Protocol

import numpy as np

from .config import Config
from .types import AudioChunk, Channel


class Vad(Protocol):
    window_samples: int

    def speech_probability(self, window: np.ndarray) -> float: ...


class VadChunker:
    def __init__(
        self,
        vad: Vad,
        config: Config,
        channel: Channel,
        window_sink: Callable[[float, np.ndarray, bool], None] | None = None,
    ) -> None:
        self._vad = vad
        self._config = config
        self._channel = channel
        self._window_sink = window_sink
        self._window_seconds = vad.window_samples / config.sample_rate
        self._buffer = np.empty(0, dtype=np.float32)
        self._next_sample_time = 0.0
        self._pre_roll: deque[tuple[float, np.ndarray]] = deque(
            maxlen=config.pre_roll_windows
        )
        self._in_speech = False
        self._chunk_windows: list[np.ndarray] = []
        self._chunk_start = 0.0
        self._first_speech_time: float | None = None
        self._last_speech_time: float | None = None
        self._silence_run = 0.0
        self._next_turn_id = 1
        self._turn_id: int | None = None

    def push(self, timestamp: float, samples: np.ndarray) -> list[AudioChunk]:
        if self._buffer.size == 0:
            self._next_sample_time = timestamp
        self._buffer = np.concatenate(
            [self._buffer, samples.astype(np.float32, copy=False)]
        )

        emitted: list[AudioChunk] = []
        window_size = self._vad.window_samples
        while self._buffer.size >= window_size:
            window = self._buffer[:window_size]
            self._buffer = self._buffer[window_size:]
            window_time = self._next_sample_time
            self._next_sample_time += self._window_seconds
            chunk = self._process_window(window_time, window)
            if chunk is not None:
                emitted.append(chunk)
        return emitted

    def flush(self) -> AudioChunk | None:
        """Finalize any in-progress chunk at end of session."""
        if not self._in_speech or not self._chunk_windows:
            return None
        return self._finalize()

    def split(self) -> AudioChunk | None:
        """Force a boundary now (speaker switch): emit the in-flight chunk
        and continue a fresh one from the same instant."""
        if not self._in_speech or not self._chunk_windows:
            return None
        return self._finalize(continue_speech=True)

    def _process_window(
        self, window_time: float, window: np.ndarray
    ) -> AudioChunk | None:
        is_speech = self._vad.speech_probability(window) >= self._config.vad_threshold
        if self._window_sink is not None:
            self._window_sink(window_time, window, is_speech)

        if not self._in_speech:
            if not is_speech:
                self._pre_roll.append((window_time, window))
                return None
            self._in_speech = True
            if self._turn_id is None:
                self._turn_id = self._next_turn_id
                self._next_turn_id += 1
            self._chunk_windows = [w for _, w in self._pre_roll]
            self._chunk_start = (
                self._pre_roll[0][0] if self._pre_roll else window_time
            )
            self._pre_roll.clear()
            self._first_speech_time = window_time
            self._last_speech_time = window_time
            self._silence_run = 0.0
            self._chunk_windows.append(window)
            return None

        self._chunk_windows.append(window)
        if is_speech:
            self._silence_run = 0.0
            self._last_speech_time = window_time
        else:
            self._silence_run += self._window_seconds

        chunk_duration = (
            window_time + self._window_seconds - self._chunk_start
        )
        if self._silence_run >= self._config.silence_split_seconds:
            return self._finalize(end_turn=True)
        if chunk_duration >= self._config.max_chunk_seconds:
            return self._finalize(continue_speech=True, end_turn=False)
        return None

    def _finalize(
        self, continue_speech: bool = False, end_turn: bool = True
    ) -> AudioChunk | None:
        samples = np.concatenate(self._chunk_windows)
        start = self._chunk_start
        end = start + samples.size / self._config.sample_rate
        turn_id = self._turn_id or 0

        speech_span = 0.0
        if self._first_speech_time is not None and self._last_speech_time is not None:
            speech_span = (
                self._last_speech_time
                + self._window_seconds
                - self._first_speech_time
            )

        if continue_speech:
            self._chunk_windows = []
            self._chunk_start = end
            self._first_speech_time = end
            self._last_speech_time = end
            self._silence_run = 0.0
            if end_turn:
                self._turn_id = self._next_turn_id
                self._next_turn_id += 1
        else:
            self._in_speech = False
            self._chunk_windows = []
            self._first_speech_time = None
            self._last_speech_time = None
            self._silence_run = 0.0
            self._turn_id = None

        if speech_span < self._config.min_speech_seconds:
            return None
        return AudioChunk(
            samples=samples,
            channel=self._channel,
            start=start,
            end=end,
            turn_id=turn_id,
            turn_end=end_turn,
            speech_seconds=speech_span,
        )
