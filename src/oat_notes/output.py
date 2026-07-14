"""Transcript formatting and the end-of-meeting file writer."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .types import Channel, TranscriptSegment

CHANNEL_LABELS = {Channel.MIC: "Microphone", Channel.LOOPBACK: "System audio"}


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.strip().lower()).strip("-")
    return slug or "meeting"


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def line_for(segment: TranscriptSegment, label_channels: bool) -> str:
    label = ""
    if segment.speaker:
        label = f"{segment.speaker}: "
    elif label_channels:
        label = f"{CHANNEL_LABELS[segment.channel]}: "
    return f"[{format_timestamp(segment.start)}] {label}{segment.text}"


class MeetingLog:
    """Unlocked by design: ``add`` runs on the transcription worker thread,
    everything else only after ``Pipeline.finish()`` has joined it."""

    def __init__(self, label_channels: bool) -> None:
        self._label_channels = label_channels
        self._segments: list[TranscriptSegment] = []

    def add(self, segment: TranscriptSegment) -> None:
        self._segments.append(segment)

    def speakers_with_lines(self) -> set[str]:
        return {
            segment.speaker for segment in self._segments if segment.speaker
        }

    def rename(self, renames: dict[str, str]) -> None:
        self._segments = [
            replace(segment, speaker=renames[segment.speaker])
            if segment.speaker in renames
            else segment
            for segment in self._segments
        ]

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
