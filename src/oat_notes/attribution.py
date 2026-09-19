"""Speaker attribution across microphone and system-audio capture."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

from .types import AudioChunk, Channel


@dataclass(frozen=True)
class Speaker:
    name: str
    speaker_id: str | None = None
    hotkey_slot: int | None = None
    active: bool = True


def parse_speakers(text: str) -> tuple[Speaker, ...]:
    """Parse a comma-separated list of names."""
    return tuple(Speaker(name=raw.strip()) for raw in text.split(",") if raw.strip())


class SwitchLog:
    """Append-only switch log for one channel; ``initial`` is active from
    t=0. Locked: writers are hotkey/UI threads, reader is the worker."""

    def __init__(self, initial: int = 0) -> None:
        self._initial = initial
        self._events: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def record(self, timestamp: float, speaker_index: int) -> None:
        with self._lock:
            self._events.append((timestamp, speaker_index))

    def active_at(self, timestamp: float) -> int:
        with self._lock:
            active = self._initial
            for event_time, speaker_index in self._events:
                if event_time > timestamp:
                    break
                active = speaker_index
            return active

    def attribute(self, start: float, end: float) -> int:
        """The speaker holding the largest share of [start, end]."""
        if end <= start:
            return self.active_at(start)
        with self._lock:
            events = list(self._events)
            active = self._initial

        overlaps: dict[int, float] = {}
        cursor = start
        for event_time, speaker_index in events:
            if event_time <= start:
                active = speaker_index
                continue
            if event_time >= end:
                break
            overlaps[active] = overlaps.get(active, 0.0) + (event_time - cursor)
            cursor = event_time
            active = speaker_index
        overlaps[active] = overlaps.get(active, 0.0) + (end - cursor)
        return max(overlaps, key=lambda speaker: overlaps[speaker])


class Attributor:
    """Names chunks from either capture source using the same roster."""

    def __init__(
        self,
        roster: tuple[Speaker, ...],
        mic_log: SwitchLog,
        loopback_log: SwitchLog,
    ) -> None:
        self._roster = list(roster)
        self._logs = {Channel.MIC: mic_log, Channel.LOOPBACK: loopback_log}

    def add(self, speaker: Speaker) -> int:
        """Append a guest speaker mid-session."""
        self._roster.append(speaker)
        return len(self._roster) - 1

    def rename(self, index: int, name: str) -> None:
        """Rename a roster member in place (mid-session naming of a guest)."""
        if 0 <= index < len(self._roster):
            self._roster[index] = replace(self._roster[index], name=name)

    def log_for(self, channel: Channel) -> SwitchLog:
        return self._logs[channel]

    def for_chunk(self, chunk: AudioChunk) -> str | None:
        if not self._roster:
            return None
        if len(self._roster) == 1:
            return self._roster[0].name
        index = self._logs[chunk.channel].attribute(chunk.start, chunk.end)
        if not 0 <= index < len(self._roster):
            index = 0
        return self._roster[index].name
