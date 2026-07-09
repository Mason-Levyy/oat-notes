"""Speaker attribution: switch log for the mic channel, channel for loopback.

Hotkey presses append (timestamp, speaker_index) to the SwitchLog. A mic
chunk is attributed to whichever speaker was active for the majority of its
span — since the VAD splits at pauses and people hand off at pauses, this
mostly self-corrects when a switch lands mid-chunk. Loopback chunks belong
to the remote party by construction; no log needed.
"""

from __future__ import annotations

import threading

from .types import AudioChunk, Channel


class SwitchLog:
    """Append-only log of speaker switches. Speaker 0 is active from t=0.

    ``record`` is called from the hotkey listener thread, ``attribute`` from
    the transcription worker — hence the lock.
    """

    def __init__(self) -> None:
        self._events: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def record(self, timestamp: float, speaker_index: int) -> None:
        with self._lock:
            self._events.append((timestamp, speaker_index))

    def active_at(self, timestamp: float) -> int:
        with self._lock:
            active = 0
            for event_time, speaker_index in self._events:
                if event_time > timestamp:
                    break
                active = speaker_index
            return active

    def attribute(self, start: float, end: float) -> int:
        """Speaker with the largest overlap with [start, end]."""
        if end <= start:
            return self.active_at(start)
        with self._lock:
            events = list(self._events)

        overlaps: dict[int, float] = {}
        active = 0
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
    """Names each chunk's speaker; returns None when there is nothing to say
    beyond the channel label (which the output formatting falls back to)."""

    def __init__(
        self,
        mic_speakers: list[str],
        switch_log: SwitchLog,
        remote_name: str | None = None,
    ) -> None:
        self._mic_speakers = mic_speakers
        self._switch_log = switch_log
        self._remote_name = remote_name

    def for_chunk(self, chunk: AudioChunk) -> str | None:
        if chunk.channel is Channel.LOOPBACK:
            return self._remote_name
        if not self._mic_speakers:
            return None
        if len(self._mic_speakers) == 1:
            return self._mic_speakers[0]
        return self._mic_speakers[self._switch_log.attribute(chunk.start, chunk.end)]
