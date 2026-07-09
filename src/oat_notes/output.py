"""Transcript formatting and the end-of-meeting file writer."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .types import Channel, TranscriptSegment

CHANNEL_LABELS = {Channel.MIC: "Me", Channel.LOOPBACK: "Remote"}


def slugify(name: str) -> str:
    """Filesystem-safe file stem: ``"Stand-up Meeting!"`` → ``stand-up-meeting``."""
    slug = re.sub(r"[^\w\-]+", "-", name.strip().lower()).strip("-")
    return slug or "meeting"


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def line_for(segment: TranscriptSegment, label_channels: bool) -> str:
    """One transcript line: ``[00:03:12] Me: Let's walk through the model.``

    ``speaker`` (Phase 2 attribution) wins over the channel label once set.
    """
    label = ""
    if segment.speaker:
        label = f"{segment.speaker}: "
    elif label_channels:
        label = f"{CHANNEL_LABELS[segment.channel]}: "
    return f"[{format_timestamp(segment.start)}] {label}{segment.text}"


class MeetingLog:
    """Accumulates segments during a session; writes the sorted transcript.

    ``add`` is called from the transcription worker thread; ``save`` only
    after ``Pipeline.finish()`` has joined it, so no locking is needed.
    """

    def __init__(self, label_channels: bool) -> None:
        self._label_channels = label_channels
        self._segments: list[TranscriptSegment] = []

    def add(self, segment: TranscriptSegment) -> None:
        self._segments.append(segment)

    @property
    def is_empty(self) -> bool:
        return not self._segments

    def save(
        self, directory: Path, started_at: datetime, name: str = "meeting"
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{slugify(name)}_{started_at:%Y-%m-%d_%H%M}"
        path = directory / f"{stem}.txt"
        ordered = sorted(self._segments, key=lambda segment: segment.start)
        lines = [line_for(segment, self._label_channels) for segment in ordered]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
