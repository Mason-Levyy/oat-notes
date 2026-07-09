"""VAD-driven chunking: turns a stream of stamped audio frames into AudioChunks.

Pure logic — the VAD is injected (anything with ``window_samples`` and
``speech_probability(window) -> float``), so tests drive it with a fake.

Splitting rules:
- a chunk opens on the first speech window, prepended with a short pre-roll
  so word onsets are not clipped;
- it closes after ``silence_split_seconds`` of continuous non-speech, or is
  force-split at ``max_chunk_seconds`` mid-speech;
- chunks whose speech span is under ``min_speech_seconds`` are dropped as
  blips.

Timestamps: ``push`` receives the capture-time stamp of each block's first
sample. Window times are derived by sample offset from the most recent block
stamp, re-anchored whenever the internal buffer drains (continuously, in live
capture), so times stay clock-anchored rather than drifting with sample
counts.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np

from .config import Config
from .types import AudioChunk, Channel


class Vad(Protocol):
    window_samples: int

    def speech_probability(self, window: np.ndarray) -> float: ...


class VadChunker:
    def __init__(self, vad: Vad, config: Config, channel: Channel) -> None:
        self._vad = vad
        self._config = config
        self._channel = channel
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

    def _process_window(
        self, window_time: float, window: np.ndarray
    ) -> AudioChunk | None:
        is_speech = self._vad.speech_probability(window) >= self._config.vad_threshold

        if not self._in_speech:
            if not is_speech:
                self._pre_roll.append((window_time, window))
                return None
            self._in_speech = True
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
            return self._finalize()
        if chunk_duration >= self._config.max_chunk_seconds:
            return self._finalize(continue_speech=True)
        return None

    def _finalize(self, continue_speech: bool = False) -> AudioChunk | None:
        samples = np.concatenate(self._chunk_windows)
        start = self._chunk_start
        end = start + samples.size / self._config.sample_rate

        speech_span = 0.0
        if self._first_speech_time is not None and self._last_speech_time is not None:
            speech_span = (
                self._last_speech_time
                + self._window_seconds
                - self._first_speech_time
            )

        if continue_speech:
            # Force-split mid-speech: the next chunk continues immediately.
            self._chunk_windows = []
            self._chunk_start = end
            self._first_speech_time = end
            self._last_speech_time = end
            self._silence_run = 0.0
        else:
            self._in_speech = False
            self._chunk_windows = []
            self._first_speech_time = None
            self._last_speech_time = None
            self._silence_run = 0.0

        if speech_span < self._config.min_speech_seconds:
            return None
        return AudioChunk(
            samples=samples, channel=self._channel, start=start, end=end
        )
