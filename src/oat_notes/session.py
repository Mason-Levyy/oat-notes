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

from .attribution import Attributor, Speaker, SwitchLog
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
    speakers: tuple[Speaker, ...] = ()
    meeting_name: str = "meeting"
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
        self._saved = False
        self.clock = SessionClock()
        self.roster: list[Speaker] = list(options.speakers)
        self.guest_indices: list[int] = []

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

        self._attributor = None
        self.active: dict[Channel, int | None] = {
            Channel.MIC: None,
            Channel.LOOPBACK: None,
        }
        mic_first = next(
            (i for i, s in enumerate(self.roster) if not s.remote), None
        )
        remote_first = next(
            (i for i, s in enumerate(self.roster) if s.remote), None
        )
        self.active[Channel.MIC] = mic_first
        self.active[Channel.LOOPBACK] = remote_first
        # Always constructed (even for an empty roster) so guests can be
        # added mid-session; channels with no members fall back to Me/Remote.
        self._attributor = Attributor(
            tuple(self.roster),
            mic_log=SwitchLog(initial=mic_first if mic_first is not None else 0),
            loopback_log=SwitchLog(
                initial=remote_first if remote_first is not None else 0
            ),
        )
        attributor = self._attributor

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
        if options.hotkeys:
            self._hotkeys = HotkeyListener(self.switch_speaker)

    def start(self) -> None:
        if self._hotkeys is not None:
            self._hotkeys.start()
        self._pipeline.start()
        for capture in self.captures:
            capture.start()

    def switch_speaker(self, index: int) -> None:
        """Route a switch to the speaker's own channel and cut its in-flight
        chunk, so the previous speaker's words transcribe immediately.

        An index past the roster auto-creates in-person guests up to that
        slot — the pressed key permanently becomes that guest's key.
        """
        if not 0 <= index < 9:
            return
        while index >= len(self.roster):
            self._create_guest(remote=False)
        channel = self._attributor.channel_of(index)
        self._attributor.log_for(index).record(self.clock.now(), index)
        self.active[channel] = index
        if channel in self.channels:
            self._pipeline.split_channel(channel)
        self._events.on_speaker(index)

    def add_guest(self, remote: bool) -> int:
        """Create a placeholder speaker mid-meeting and switch to them.

        The name is backfilled at save time via ``renames``.
        """
        index = self._create_guest(remote)
        self.switch_speaker(index)
        return index

    def _create_guest(self, remote: bool) -> int:
        guest = Speaker(f"Guest {len(self.guest_indices) + 1}", remote=remote)
        index = self._attributor.add(guest)
        self.roster.append(guest)
        self.guest_indices.append(index)
        return index

    def elapsed(self) -> float:
        return self.clock.now()

    @property
    def dropped_blocks(self) -> int:
        return sum(capture.dropped_blocks for capture in self.captures)

    def stop(self) -> None:
        """Stop capture and drain the pipeline. Call ``save`` afterwards —
        it's separate so guest names can be backfilled first."""
        if self._stopped:
            return
        self._stopped = True
        if self._hotkeys is not None:
            self._hotkeys.stop()
        for capture in self.captures:
            capture.stop()
        self._pipeline.finish()
        self._pa.terminate()

    def save(self, renames: dict[str, str] | None = None) -> Path | None:
        """Apply guest-name backfills and write the transcript file."""
        if self._saved or not self._options.save_file or self.log.is_empty:
            return None
        if renames:
            self.log.rename(
                {old: new.strip() for old, new in renames.items() if new.strip()}
            )
        self._saved = True
        return self.log.save(
            self._options.out_dir, self._started_at, self._options.meeting_name
        )

    def _handle_segment(self, segment: TranscriptSegment, latency: float) -> None:
        self.log.add(segment)
        if segment.speaker:
            label = segment.speaker
        elif self._label_channels:
            label = CHANNEL_LABELS[segment.channel]
        else:
            label = ""
        self._events.on_segment(segment, label, latency)
