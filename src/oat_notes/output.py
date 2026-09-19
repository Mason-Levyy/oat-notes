"""Transcript formatting and the end-of-meeting file writer."""

from __future__ import annotations

import re
import sys
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .types import Channel, TranscriptSegment

CHANNEL_LABELS = {Channel.MIC: "Microphone", Channel.LOOPBACK: "System audio"}
JOURNAL_SUFFIX = ".partial.txt"


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.strip().lower()).strip("-")
    return slug or "meeting"


def _stem_for(started_at: datetime, name: str) -> str:
    return f"{slugify(name)}_{started_at:%Y-%m-%d_%H%M}"


class TranscriptJournal:
    """Append-only crash-safety log for one meeting. Each finalized line is
    flushed to disk as it lands so a crash or power loss keeps whatever was
    said; the file is deleted on a clean save. Written from both the
    transcription worker (lines) and the UI thread (notes), so appends are
    locked."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._handle = None

    def append(self, line: str) -> None:
        with self._lock:
            if self._handle is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self._path.open("a", encoding="utf-8")
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self, delete: bool = False) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            if delete:
                self._path.unlink(missing_ok=True)


def journal_path_for(directory: Path, started_at: datetime, name: str) -> Path:
    return directory / f"{_stem_for(started_at, name)}{JOURNAL_SUFFIX}"


def recover_journals(directory: Path) -> list[Path]:
    """Turn journals orphaned by a crashed prior run into recovered
    transcripts. Empty journals are just cleaned up. Returns the paths of
    the recovered transcript files.

    A journal another process still holds open is not orphaned — it belongs
    to a live run — so it is left where it is. Startup must survive that:
    recovery is a courtesy, and no journal is worth refusing to launch over.
    """
    if not directory.exists():
        return []
    recovered: list[Path] = []
    for partial in sorted(directory.glob("*" + JOURNAL_SUFFIX)):
        try:
            if partial.stat().st_size == 0:
                partial.unlink(missing_ok=True)
                continue
            stem = partial.name[: -len(JOURNAL_SUFFIX)]
            target = directory / f"{stem}_recovered.txt"
            counter = 1
            while target.exists():
                counter += 1
                target = directory / f"{stem}_recovered_{counter}.txt"
            partial.rename(target)
        except OSError as error:
            print(f"warning: leaving {partial.name} in place: {error}", file=sys.stderr)
            continue
        recovered.append(target)
    return recovered


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

    def __init__(
        self, label_channels: bool, journal: TranscriptJournal | None = None
    ) -> None:
        self._label_channels = label_channels
        self._segments: list[TranscriptSegment] = []
        self._notes: list[tuple[float, str]] = []
        self._journal = journal

    def add(self, segment: TranscriptSegment) -> int:
        """Append a segment and return its line id for later cleanup."""
        self._segments.append(segment)
        if self._journal is not None:
            self._journal.append(line_for(segment, self._label_channels))
        return len(self._segments) - 1

    def apply_cleanup(self, results: dict[int, str | None]) -> None:
        """Rewrite (or drop, for ``None``) lines by id. Call only after the
        pipeline and cleanup worker have finished — same single-threaded
        window as ``save``."""
        kept: list[TranscriptSegment] = []
        for index, segment in enumerate(self._segments):
            if index not in results:
                kept.append(segment)
                continue
            text = results[index]
            if text is not None:
                kept.append(replace(segment, text=text))
        self._segments = kept

    def add_note(self, timestamp: float, text: str) -> None:
        self._notes.append((timestamp, text))
        if self._journal is not None:
            self._journal.append(f"[{format_timestamp(timestamp)}] NOTE: {text}")

    def speakers_with_lines(self) -> set[str]:
        return {
            segment.speaker for segment in self._segments if segment.speaker
        }

    def has_line(self, line_id: int) -> bool:
        return 0 <= line_id < len(self._segments)

    def relabel(self, line_id: int, speaker: str) -> bool:
        """Move one line to a different speaker, by id.

        ``rename`` is keyed by name, so it cannot express this: every
        unattributed line shares the name "Unknown" and renaming it would move
        all of them at once.

        The append-only journal keeps whatever it already flushed — it is a
        crash artifact, not the transcript. ``save`` rebuilds from
        ``_segments``, so the correction lands in the file that matters.
        """
        if not self.has_line(line_id):
            return False
        self._segments[line_id] = replace(self._segments[line_id], speaker=speaker)
        return True

    def rename(self, renames: dict[str, str]) -> None:
        self._segments = [
            replace(segment, speaker=renames[segment.speaker])
            if segment.speaker in renames
            else segment
            for segment in self._segments
        ]

    @property
    def is_empty(self) -> bool:
        return not self._segments and not self._notes

    def discard_journal(self) -> None:
        """Drop the crash-safety journal without saving (meeting discarded)."""
        if self._journal is not None:
            self._journal.close(delete=True)

    def save(
        self, directory: Path, started_at: datetime, name: str = "meeting"
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stem = _stem_for(started_at, name)
        path = directory / f"{stem}.txt"
        entries = [
            (segment.start, line_for(segment, self._label_channels))
            for segment in self._segments
        ] + [
            (timestamp, f"[{format_timestamp(timestamp)}] NOTE: {text}")
            for timestamp, text in self._notes
        ]
        entries.sort(key=lambda entry: entry[0])
        lines = [line for _, line in entries]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self._journal is not None:
            self._journal.close(delete=True)
        return path
