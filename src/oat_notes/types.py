"""Core value types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Channel(Enum):
    """Which capture stream a piece of audio came from."""

    MIC = "mic"
    LOOPBACK = "loopback"


@dataclass(frozen=True)
class AudioChunk:
    """A VAD-delimited span of speech from one channel.

    ``samples`` is float32 mono at the configured sample rate. ``start`` and
    ``end`` are seconds from the session epoch, stamped at capture time.
    """

    samples: np.ndarray
    channel: Channel
    start: float
    end: float
    turn_id: int = 0
    turn_end: bool = True
    manual_speaker_index: int | None = None
    speech_seconds: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class TranscriptSegment:
    """Transcribed text for one chunk, with whatever attribution was decided."""

    text: str
    channel: Channel
    start: float
    end: float
    speaker: str | None = None
    speaker_id: str | None = None
    speaker_index: int | None = None
    attribution: str | None = None
    confidence: float | None = None
    turn_end: bool = True
