"""VadChunker state-machine tests, driven by a scripted fake VAD."""

import numpy as np
import pytest

from oat_notes.chunker import VadChunker
from oat_notes.config import Config
from oat_notes.types import Channel

CONFIG = Config()
WINDOW = CONFIG.vad_window_samples
WINDOW_SECONDS = WINDOW / CONFIG.sample_rate  # 0.032 s
SILENCE_WINDOWS = 16  # first count where silence_run (n * 0.032) >= 0.5


class FakeVad:
    window_samples = WINDOW

    def __init__(self, probabilities):
        self._probabilities = list(probabilities)
        self._position = 0

    def speech_probability(self, window):
        probability = self._probabilities[self._position]
        self._position += 1
        return probability


def make_chunker(probabilities):
    return VadChunker(FakeVad(probabilities), CONFIG, Channel.MIC)


def windows(count):
    return np.zeros(count * WINDOW, dtype=np.float32)


def test_silence_split_emits_chunk():
    idle, speech = 5, 31
    chunker = make_chunker([0.0] * idle + [1.0] * speech + [0.0] * SILENCE_WINDOWS)
    chunks = chunker.push(0.0, windows(idle + speech + SILENCE_WINDOWS))

    assert len(chunks) == 1
    chunk = chunks[0]
    pre_roll_start = idle - CONFIG.pre_roll_windows
    total_windows = CONFIG.pre_roll_windows + speech + SILENCE_WINDOWS
    assert chunk.start == pytest.approx(pre_roll_start * WINDOW_SECONDS)
    assert chunk.end == pytest.approx(chunk.start + total_windows * WINDOW_SECONDS)
    assert chunk.samples.size == total_windows * WINDOW
    assert chunk.channel is Channel.MIC


def test_blip_dropped_but_real_speech_kept():
    blip, speech = 3, 31
    chunker = make_chunker(
        [1.0] * blip
        + [0.0] * SILENCE_WINDOWS
        + [1.0] * speech
        + [0.0] * SILENCE_WINDOWS
    )
    chunks = chunker.push(0.0, windows(blip + speech + 2 * SILENCE_WINDOWS))

    assert len(chunks) == 1  # the 3-window blip (0.096 s of speech) was dropped
    assert chunks[0].samples.size >= speech * WINDOW


def test_force_split_at_max_chunk_length():
    total = 600  # 19.2 s of continuous speech
    chunker = make_chunker([1.0] * total)
    chunks = chunker.push(0.0, windows(total))
    final = chunker.flush()

    assert len(chunks) == 1
    assert final is not None
    first, second = chunks[0], final
    assert first.duration >= CONFIG.max_chunk_seconds
    assert first.duration == pytest.approx(469 * WINDOW_SECONDS)
    assert second.start == pytest.approx(first.end)
    assert first.samples.size + second.samples.size == total * WINDOW


def test_timestamps_reanchor_on_block_stamp():
    chunker = make_chunker(
        [0.0] * 4 + [0.0] * 2 + [1.0] * 10 + [0.0] * SILENCE_WINDOWS
    )
    assert chunker.push(0.0, windows(4)) == []

    block_start = 10.0
    chunks = chunker.push(block_start, windows(2 + 10 + SILENCE_WINDOWS))
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start == pytest.approx(block_start)  # pre-roll from the new block
    total_windows = CONFIG.pre_roll_windows + 10 + SILENCE_WINDOWS
    assert chunk.end == pytest.approx(block_start + total_windows * WINDOW_SECONDS)


def test_flush_returns_in_progress_chunk():
    chunker = make_chunker([1.0] * 20)
    assert chunker.push(0.0, windows(20)) == []

    chunk = chunker.flush()
    assert chunk is not None
    assert chunk.start == pytest.approx(0.0)
    assert chunk.samples.size == 20 * WINDOW
    assert chunker.flush() is None


def test_split_mid_speech_emits_and_continues():
    chunker = make_chunker([1.0] * 62 + [0.0] * SILENCE_WINDOWS)
    assert chunker.push(0.0, windows(31)) == []

    forced = chunker.split()
    assert forced is not None
    assert forced.samples.size == 31 * WINDOW

    rest = chunker.push(31 * WINDOW_SECONDS, windows(31 + SILENCE_WINDOWS))
    assert len(rest) == 1
    assert rest[0].start == pytest.approx(forced.end)


def test_split_when_idle_returns_none():
    chunker = make_chunker([0.0] * 5)
    chunker.push(0.0, windows(5))
    assert chunker.split() is None


def test_split_twice_returns_none_second_time():
    chunker = make_chunker([1.0] * 31)
    chunker.push(0.0, windows(31))
    assert chunker.split() is not None
    assert chunker.split() is None


def test_flush_when_idle_returns_none():
    chunker = make_chunker([0.0] * 10)
    chunker.push(0.0, windows(10))
    assert chunker.flush() is None


def test_partial_windows_buffer_across_pushes():
    speech = [1.0] * 31 + [0.0] * SILENCE_WINDOWS
    chunker = make_chunker(speech)
    half = WINDOW // 2
    emitted = []
    total_samples = (31 + SILENCE_WINDOWS) * WINDOW
    for offset in range(0, total_samples, half):
        emitted += chunker.push(
            offset / CONFIG.sample_rate, np.zeros(half, dtype=np.float32)
        )
    assert len(emitted) == 1
