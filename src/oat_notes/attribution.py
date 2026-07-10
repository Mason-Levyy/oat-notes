"""Speaker attribution: one roster, but each channel (mic/loopback) has its
own switch log and only its own members may claim its chunks."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .types import AudioChunk, Channel


@dataclass(frozen=True)
class Speaker:
    name: str
    remote: bool = False


def parse_speakers(text: str) -> tuple[Speaker, ...]:
    """``"Mason, Sarah, Priya*, Dev*"`` — a ``*`` suffix marks remote."""
    roster = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        remote = raw.endswith("*")
        roster.append(Speaker(name=raw.rstrip("*").strip(), remote=remote))
    return tuple(roster)


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
    """Names each chunk's speaker; None when the chunk's channel has nobody
    listed, letting output fall back to the Me/Remote channel label."""

    def __init__(
        self,
        roster: tuple[Speaker, ...],
        mic_log: SwitchLog,
        loopback_log: SwitchLog,
    ) -> None:
        self._roster = list(roster)
        self._logs = {Channel.MIC: mic_log, Channel.LOOPBACK: loopback_log}
        self._members = {
            Channel.MIC: [
                index for index, speaker in enumerate(roster) if not speaker.remote
            ],
            Channel.LOOPBACK: [
                index for index, speaker in enumerate(roster) if speaker.remote
            ],
        }

    def add(self, speaker: Speaker) -> int:
        """Append a speaker mid-session (walk-in / call-in guest)."""
        self._roster.append(speaker)
        index = len(self._roster) - 1
        self._members[self.channel_of(index)].append(index)
        return index

    def channel_of(self, index: int) -> Channel:
        return Channel.LOOPBACK if self._roster[index].remote else Channel.MIC

    def log_for(self, index: int) -> SwitchLog:
        return self._logs[self.channel_of(index)]

    def first_member(self, channel: Channel) -> int | None:
        members = self._members[channel]
        return members[0] if members else None

    def for_chunk(self, chunk: AudioChunk) -> str | None:
        members = self._members[chunk.channel]
        if not members:
            return None
        if len(members) == 1:
            return self._roster[members[0]].name
        index = self._logs[chunk.channel].attribute(chunk.start, chunk.end)
        if index not in members:
            index = members[0]
        return self._roster[index].name
