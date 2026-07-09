"""One live meeting session: capture + pipeline + attribution + log.

Both front ends drive this class — the console (cli.py) and the web UI
(server.py). Speaker switches arrive from the global hotkeys or the UI and
funnel through ``switch_speaker``, which records the switch and cuts the
in-flight mic chunk so the handoff attributes exactly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .attribution import Attributor, SwitchLog
from .capture import AudioCapture, find_default_loopback
from .clock import SessionClock
from .config import Config
from .hotkeys import HotkeyListener
from .output import CHANNEL_LABELS, MeetingLog
from .pipeline import Pipeline
from .transcriber import Transcriber
from .types import Channel, TranscriptSegment


@dataclass(frozen=True)
class SessionOptions:
    speakers: tuple[str, ...] = ()
    remote_name: str | None = None
    use_loopback: bool = True
    device_index: int | None = None
    loopback_index: int | None = None
    out_dir: Path = Path("transcripts")
    save_file: bool = True
    hotkeys: bool = True


@dataclass(frozen=True)
class SessionEvents:
    """Callbacks fire on pipeline/hotkey threads — keep them quick."""

    on_segment: Callable[[TranscriptSegment, str, float], None] = (
        lambda segment, label, latency: None
    )
    on_speaker: Callable[[int], None] = lambda index: None


class Session:
    def __init__(
        self,
        config: Config,
        options: SessionOptions,
        transcriber: Transcriber,
        events: SessionEvents = SessionEvents(),
    ) -> None:
        import pyaudiowpatch as pyaudio

        self._config = config
        self._options = options
        self._events = events
        self._started_at = datetime.now()
        self._stopped = False
        self.clock = SessionClock()
        self.switch_log = SwitchLog()
        self.active_speaker = 0

        self._pa = pyaudio.PyAudio()
        loopback_info = None
        if options.use_loopback:
            if options.loopback_index is not None:
                loopback_info = self._pa.get_device_info_by_index(
                    options.loopback_index
                )
            else:
                loopback_info = find_default_loopback(self._pa)
                if loopback_info is None:
                    print(
                        "warning: no loopback device found — capturing mic only",
                        file=sys.stderr,
                    )
        self.channels = (
            (Channel.MIC, Channel.LOOPBACK) if loopback_info else (Channel.MIC,)
        )
        self._label_channels = len(self.channels) > 1

        attributor = None
        if options.speakers or options.remote_name:
            attributor = Attributor(
                list(options.speakers), self.switch_log, options.remote_name
            )

        self.log = MeetingLog(label_channels=self._label_channels)
        self._pipeline = Pipeline(
            config,
            self.clock,
            transcriber,
            self._handle_segment,
            channels=self.channels,
            attributor=attributor,
        )

        self.captures = [
            AudioCapture(
                self._pa, options.device_index, Channel.MIC, self.clock, config,
                self._pipeline.frame_queue,
            )
        ]
        if loopback_info is not None:
            self.captures.append(
                AudioCapture(
                    self._pa, int(loopback_info["index"]), Channel.LOOPBACK,
                    self.clock, config, self._pipeline.frame_queue,
                )
            )

        self._hotkeys = None
        if options.hotkeys and len(options.speakers) > 1:
            self._hotkeys = HotkeyListener(
                len(options.speakers), self.switch_speaker
            )

    def start(self) -> None:
        if self._hotkeys is not None:
            self._hotkeys.start()
        self._pipeline.start()
        for capture in self.captures:
            capture.start()

    def switch_speaker(self, index: int) -> None:
        """Record a speaker switch and cut the in-flight mic chunk."""
        if not 0 <= index < len(self._options.speakers):
            return
        self.switch_log.record(self.clock.now(), index)
        self.active_speaker = index
        self._pipeline.split_channel(Channel.MIC)
        self._events.on_speaker(index)

    def elapsed(self) -> float:
        return self.clock.now()

    @property
    def dropped_blocks(self) -> int:
        return sum(capture.dropped_blocks for capture in self.captures)

    def stop(self) -> Path | None:
        """Stop capture, drain the pipeline, save and return the transcript."""
        if self._stopped:
            return None
        self._stopped = True
        if self._hotkeys is not None:
            self._hotkeys.stop()
        for capture in self.captures:
            capture.stop()
        self._pipeline.finish()
        self._pa.terminate()
        if not self._options.save_file or self.log.is_empty:
            return None
        return self.log.save(self._options.out_dir, self._started_at)

    def _handle_segment(self, segment: TranscriptSegment, latency: float) -> None:
        self.log.add(segment)
        if segment.speaker:
            label = segment.speaker
        elif self._label_channels:
            label = CHANNEL_LABELS[segment.channel]
        else:
            label = ""
        self._events.on_segment(segment, label, latency)
