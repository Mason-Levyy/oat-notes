"""One live meeting session, shared by the console and web front ends."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from .attribution import Attributor, Speaker, SwitchLog
from .capture import AudioCapture, find_default_loopback
from .cleanup import CleanupWorker, LineCleaner
from .clock import SessionClock
from .config import Config
from .hotkeys import HotkeyListener
from .output import (
    CHANNEL_LABELS,
    MeetingLog,
    TranscriptJournal,
    format_timestamp,
    journal_path_for,
)
from .pipeline import Pipeline, ProfileLearningUpdate
from .speaker_id import SpeakerEmbeddingEngine, SpeakerResolver
from .speaker_store import SpeakerProfile, SpeakerStore
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
    hotkey_modifiers: tuple[str, ...] = ("ctrl", "alt")
    hotkey_listener: HotkeyListener | None = None
    speaker_store: SpeakerStore | None = None
    embedding_engine: SpeakerEmbeddingEngine | None = None
    tracking_embedding_engine: SpeakerEmbeddingEngine | None = None
    cleaner: LineCleaner | None = None


@dataclass(frozen=True)
class SessionEvents:
    """Callbacks fire on pipeline/hotkey threads — keep them quick."""

    on_segment: Callable[[TranscriptSegment, str, float, int], None] = (
        lambda segment, label, latency, line_id: None
    )
    on_line_cleaned: Callable[[int, "str | None"], None] = (
        lambda line_id, text: None
    )
    on_speaker: Callable[[int], None] = lambda index: None
    on_attribution: Callable[[int | None, Channel, str, float | None], None] = (
        lambda index, channel, source, confidence: None
    )
    on_hotkey_bank: Callable[[int], None] = lambda bank: None
    on_profile_learning: Callable[[ProfileLearningUpdate], None] = lambda update: None
    on_note: Callable[[dict], None] = lambda note: None


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
        self.roster = self._assign_hotkey_slots(options.speakers)
        self.guest_indices: list[int] = []
        self.hotkey_bank = 0

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

        self.active: dict[Channel, int | None] = {
            Channel.MIC: None,
            Channel.LOOPBACK: None,
        }
        # ``active`` remains channel-specific for attribution, but the UI needs
        # one unambiguous person to highlight.  Start with nobody highlighted
        # until a manual selection or a live speaker match arrives.
        self.current_speaker: int | None = None
        self._current_speaker_channel: Channel | None = None
        self.profile_learning: dict | None = None
        self._manual_override_pending = {
            Channel.MIC: False,
            Channel.LOOPBACK: False,
        }
        first = 0 if self.roster else None
        self.active[Channel.MIC] = first
        self.active[Channel.LOOPBACK] = first
        # Constructed even for an empty roster: guests can be added mid-session.
        self._attributor = Attributor(
            tuple(self.roster),
            mic_log=SwitchLog(initial=first if first is not None else 0),
            loopback_log=SwitchLog(initial=first if first is not None else 0),
        )
        self._speaker_resolver = (
            SpeakerResolver(
                self.roster,
                options.speaker_store,
                options.embedding_engine,
                config.sample_rate,
                tracking_engine=options.tracking_embedding_engine,
            )
            if options.speaker_store is not None
            else None
        )

        journal = (
            TranscriptJournal(
                journal_path_for(
                    options.out_dir, self._started_at, options.meeting_name
                )
            )
            if options.save_file
            else None
        )
        self.log = MeetingLog(label_channels=self._label_channels, journal=journal)
        self._cleanup = (
            CleanupWorker(
                options.cleaner,
                self._handle_line_cleaned,
                context_before=config.cleanup_context_before,
                context_after=config.cleanup_context_after,
                max_wait_seconds=config.cleanup_max_wait_seconds,
            )
            if options.cleaner is not None
            else None
        )
        self._pipeline = Pipeline(
            config,
            self.clock,
            transcriber,
            self._handle_segment,
            channels=self.channels,
            attributor=self._attributor,
            speaker_resolver=self._speaker_resolver,
            speaker_tracking_sink=self._handle_tracking_attribution,
            profile_learning_sink=self._handle_profile_learning,
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
        self._owns_hotkeys = False
        if options.hotkeys:
            self._hotkeys = options.hotkey_listener
            if self._hotkeys is None:
                self._hotkeys = HotkeyListener()
                self._owns_hotkeys = True

    @staticmethod
    def _assign_hotkey_slots(speakers: tuple[Speaker, ...]) -> list[Speaker]:
        roster: list[Speaker] = []
        used: set[int] = set()
        next_slot = 0
        for speaker in speakers:
            requested = speaker.hotkey_slot
            if requested is None or requested < 0 or requested in used:
                while next_slot in used:
                    next_slot += 1
                requested = next_slot
            used.add(requested)
            next_slot = max(next_slot, requested + 1)
            roster.append(replace(speaker, hotkey_slot=requested))
        return roster

    def start(self) -> None:
        if self._hotkeys is not None:
            self._hotkeys.bind_speaker_switch(
                self._options.hotkey_modifiers, self.switch_hotkey, self.page_hotkeys
            )
            if self._owns_hotkeys:
                self._hotkeys.start()
        if self._cleanup is not None:
            self._cleanup.start()
        self._pipeline.start()
        for capture in self.captures:
            capture.start()

    def switch_speaker(self, index: int) -> None:
        """Select a person for either audio source and cut in-flight chunks."""
        if not 0 <= index < len(self.roster) or not self.roster[index].active:
            return
        self.current_speaker = index
        self._current_speaker_channel = None
        timestamp = self.clock.now()
        for channel in self.channels:
            self._attributor.log_for(channel).record(timestamp, index)
            self.active[channel] = index
            if self._speaker_resolver is not None:
                self._manual_override_pending[channel] = True
                self._pipeline.manual_override(channel, index)
            else:
                self._pipeline.split_channel(channel)
        if self._speaker_resolver is not None:
            self._pipeline.begin_profile_learning(index, timestamp)
        self._events.on_speaker(index)

    def switch_hotkey(self, offset: int) -> None:
        slot = self.hotkey_bank * 9 + offset
        index = next(
            (
                item
                for item, speaker in enumerate(self.roster)
                if speaker.hotkey_slot == slot and speaker.active
            ),
            None,
        )
        if index is None:
            index = self._create_guest(hotkey_slot=slot)
        self.switch_speaker(index)

    def page_hotkeys(self, direction: int) -> None:
        # Banks are intentionally unbounded in the forward direction. This
        # lets a user page into a completely empty bank and create a Guest at
        # any exact slot with the next digit press.
        self.hotkey_bank = max(0, self.hotkey_bank + direction)
        self._events.on_hotkey_bank(self.hotkey_bank)

    def add_guest(self) -> int:
        """Create a placeholder speaker mid-meeting and switch to them.

        The name is backfilled at save time via ``renames``.
        """
        used = {speaker.hotkey_slot for speaker in self.roster}
        start = self.hotkey_bank * 9
        slot = next((item for item in range(start, start + 9) if item not in used), None)
        if slot is None:
            slot = max((item for item in used if item is not None), default=-1) + 1
        index = self._create_guest(hotkey_slot=slot)
        self.switch_speaker(index)
        return index

    def _create_guest(self, hotkey_slot: int | None = None) -> int:
        guest = Speaker(
            f"Guest {len(self.guest_indices) + 1}",
            hotkey_slot=hotkey_slot,
        )
        index = self._attributor.add(guest)
        self.roster.append(guest)
        self.guest_indices.append(index)
        return index

    def add_speaker(self, name: str, speaker_id: str | None = None) -> int:
        """Add a known person to the live roster without switching to them."""
        used = {speaker.hotkey_slot for speaker in self.roster}
        start = self.hotkey_bank * 9
        slot = next((item for item in range(start, start + 9) if item not in used), None)
        if slot is None:
            slot = max((item for item in used if item is not None), default=-1) + 1
        speaker = Speaker(name, speaker_id=speaker_id, hotkey_slot=slot)
        index = self._attributor.add(speaker)
        self.roster.append(speaker)
        return index

    def remove_speaker(self, index: int) -> str | None:
        """Deactivate a roster member who hasn't spoken yet — including one
        that's currently "active" because it was just created (a mis-clicked
        Guest) or manually selected but never actually spoke.

        Returns an error message, or ``None`` on success. Roster position is
        never reused for a different person mid-session — segments, hotkey
        state, and in-flight pipeline threads all reference it by index — so
        removal only flips ``active`` rather than shrinking the list.
        """
        if not 0 <= index < len(self.roster):
            return "speaker not found"
        speaker = self.roster[index]
        if not speaker.active:
            return "speaker already removed"
        if self.profile_learning and self.profile_learning.get("speaker_index") == index:
            return "a voice sample is being captured for this speaker"
        if speaker.name in self.log.speakers_with_lines():
            return "speaker has already spoken"
        self.roster[index] = replace(speaker, active=False)
        if self.current_speaker == index:
            self.current_speaker = None
            self._current_speaker_channel = None
        for channel, active_index in list(self.active.items()):
            if active_index == index:
                self.active[channel] = None
        return None

    def rename_speaker(self, index: int, name: str) -> str | None:
        """Rename a roster member mid-meeting (e.g. name a Guest once you
        know who they are). Renames past lines and, for an enrolled speaker,
        keeps the saved library profile name in sync. Returns an error
        message or ``None`` on success."""
        name = name.strip()
        if not 0 <= index < len(self.roster):
            return "speaker not found"
        if not name:
            return "a name is required"
        speaker = self.roster[index]
        old = speaker.name
        if old == name:
            return None
        self.roster[index] = replace(speaker, name=name)
        self._attributor.rename(index, name)
        self.log.rename({old: name})
        if speaker.speaker_id and self._options.speaker_store is not None:
            try:
                self._options.speaker_store.rename_speaker(speaker.speaker_id, name)
            except (KeyError, ValueError):
                pass
        return None

    def cancel_profile_learning(self) -> None:
        self._pipeline.cancel_profile_learning()

    def add_note(self, text: str) -> dict:
        timestamp = self.clock.now()
        self.log.add_note(timestamp, text)
        note = {"time": format_timestamp(timestamp), "text": text}
        self._events.on_note(note)
        return note

    def profile_status(self, index: int) -> tuple[str, float]:
        if self._speaker_resolver is None:
            return "untrained", 0.0
        return self._speaker_resolver.profile_status(index)

    def persist_guest(self, guest_name: str, speaker_id: str) -> SpeakerProfile | None:
        if self._speaker_resolver is None:
            return None
        index = next(
            (
                item
                for item in self.guest_indices
                if self.roster[item].name == guest_name
            ),
            None,
        )
        if index is None:
            return None
        return self._speaker_resolver.persist_guest(index, speaker_id)

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
            self._hotkeys.unbind_speaker_switch()
            if self._owns_hotkeys:
                self._hotkeys.stop()
        for capture in self.captures:
            capture.stop()
        self._pipeline.finish()
        if self._cleanup is not None:
            self._cleanup.finish()
        self._pa.terminate()

    def discard(self) -> None:
        """Drop the meeting without saving — remove its crash-safety journal
        so it isn't recovered on the next launch."""
        self.log.discard_journal()

    def save(self, renames: dict[str, str] | None = None) -> Path | None:
        """Apply guest-name backfills and write the transcript file."""
        if self._saved or not self._options.save_file or self.log.is_empty:
            return None
        if self._cleanup is not None:
            self.log.apply_cleanup(self._cleanup.results())
        if renames:
            self.log.rename(
                {old: new.strip() for old, new in renames.items() if new.strip()}
            )
        self._saved = True
        return self.log.save(
            self._options.out_dir, self._started_at, self._options.meeting_name
        )

    def _handle_segment(self, segment: TranscriptSegment, latency: float) -> None:
        # A long utterance may produce several max-duration transcript
        # chunks. Keep its turn connected; rolling tracking updates the live
        # chip independently while turn attribution closes on natural silence.
        pending_manual = self._manual_override_pending.get(segment.channel, False)
        if segment.attribution == "manual" and segment.speaker_index is not None:
            self.active[segment.channel] = segment.speaker_index
            self.current_speaker = segment.speaker_index
            self._current_speaker_channel = segment.channel
            self._events.on_attribution(
                segment.speaker_index,
                segment.channel,
                "manual",
                segment.confidence,
            )
        elif (
            segment.turn_end
            and segment.attribution == "unknown"
            and not pending_manual
        ):
            self.active[segment.channel] = None
            if self._current_speaker_channel == segment.channel:
                self.current_speaker = None
                self._current_speaker_channel = None
            self._events.on_attribution(
                None, segment.channel, "unknown", segment.confidence
            )
        elif (
            segment.turn_end
            and segment.speaker_index is not None
            and not pending_manual
        ):
            self.active[segment.channel] = segment.speaker_index
            self.current_speaker = segment.speaker_index
            self._current_speaker_channel = segment.channel
            self._events.on_attribution(
                segment.speaker_index,
                segment.channel,
                segment.attribution or "manual",
                segment.confidence,
            )
        if segment.attribution == "manual":
            self._manual_override_pending[segment.channel] = False
        if not segment.text:
            return
        line_id = self.log.add(segment)
        if self._cleanup is not None:
            self._cleanup.submit(line_id, segment.text)
        if segment.speaker:
            label = segment.speaker
        elif self._label_channels:
            label = CHANNEL_LABELS[segment.channel]
        else:
            label = ""
        self._events.on_segment(segment, label, latency, line_id)

    def _handle_tracking_attribution(
        self,
        index: int,
        channel: Channel,
        source: str,
        confidence: float | None,
    ) -> None:
        """Publish stable rolling matches without waiting for a VAD turn end."""
        if self._manual_override_pending.get(channel, False):
            return
        self.active[channel] = index
        self.current_speaker = index
        self._current_speaker_channel = channel
        self._events.on_attribution(index, channel, source, confidence)

    def _handle_profile_learning(self, update: ProfileLearningUpdate) -> None:
        self.profile_learning = update.to_dict() if update.phase == "collecting" else None
        self._events.on_profile_learning(update)

    def _handle_line_cleaned(self, line_id: int, text: str | None) -> None:
        """Fires on the cleanup worker thread — forward only."""
        self._events.on_line_cleaned(line_id, text)
