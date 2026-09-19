"""Transcript lines and notes as the browser sees them.

Appended from the pipeline and cleanup threads, relabelled from HTTP and
hotkey threads, and snapshotted for every ``/api/state`` reply — so every
touch goes through one lock. Nothing here calls out while holding it."""

from __future__ import annotations

import threading
from typing import Any

Line = dict[str, Any]


class LineFeed:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: list[Line] = []
        self._notes: list[Line] = []

    def reset(self) -> None:
        with self._lock:
            self._lines = []
            self._notes = []

    def append(self, line: Line) -> None:
        with self._lock:
            self._lines.append(line)

    def add_note(self, note: Line) -> None:
        with self._lock:
            self._notes.append(note)

    def drop(self, line_id: int) -> bool:
        with self._lock:
            position = self._position(line_id)
            if position is None:
                return False
            del self._lines[position]
            return True

    def update(self, line_id: int, **fields: Any) -> tuple[Line, Line] | None:
        """Merge ``fields`` into one line; returns copies of the line before
        and after, or None when no line has that id."""
        with self._lock:
            position = self._position(line_id)
            if position is None:
                return None
            before = dict(self._lines[position])
            self._lines[position].update(fields)
            return before, dict(self._lines[position])

    def relabel(self, renames: dict[str, str]) -> None:
        with self._lock:
            for line in self._lines:
                label = line.get("label")
                if label in renames:
                    line["label"] = renames[label]

    def snapshot(self) -> tuple[list[Line], list[Line]]:
        with self._lock:
            return [dict(line) for line in self._lines], [dict(note) for note in self._notes]

    def _position(self, line_id: int) -> int | None:
        for position, line in enumerate(self._lines):
            if line.get("id") == line_id:
                return position
        return None
