"""Localhost-only web UI: stdlib HTTP + SSE, one meeting at a time. Each
connected SSE client holds a handler thread."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import webbrowser
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Callable

from .attribution import Speaker
from .config import Config
from .dictation.controller import DictationController, DictationOptions
from .hotkeys import HotkeyListener
from .output import format_timestamp, recover_journals
from .overlay import Overlay
from .settings import AppSettings, SettingsStore
from .speaker_store import SpeakerStore
from .types import Channel, TranscriptSegment


def dictation_options(settings: AppSettings) -> DictationOptions:
    """The single place persisted settings become runtime dictation options."""
    return DictationOptions(
        enabled=settings.dictation_enabled,
        modifiers=settings.dictation_modifiers,
        email_modifiers=settings.dictation_email_modifiers,
        replay_modifiers=settings.dictation_replay_modifiers,
        tap_seconds=settings.dictation_tap_ms / 1000.0,
        restore_clipboard=settings.dictation_restore_clipboard,
        email_detection=settings.dictation_email_detection,
        spoken_punctuation=settings.dictation_spoken_punctuation,
        vocabulary=settings.dictation_vocabulary,
    )


class EventHub:
    """Fans JSON events out to every connected SSE client."""

    def __init__(self) -> None:
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=512)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, event: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass


class AppState:
    """Everything the handlers share. One meeting at a time."""

    def __init__(
        self,
        config: Config,
        transcriber,
        out_dir: Path = Path("transcripts"),
        settings: AppSettings = AppSettings(),
        save_settings: Callable[[AppSettings], None] | None = None,
        speaker_store: SpeakerStore | None = None,
        embedding_engine=None,
        tracking_embedding_engine=None,
        hotkey_listener=None,
        dictation=None,
    ) -> None:
        self.config = config
        self.transcriber = transcriber
        self._out_dir = out_dir
        self.settings = settings
        self._save_settings = save_settings
        self.speaker_store = speaker_store
        self.embedding_engine = embedding_engine
        self.tracking_embedding_engine = tracking_embedding_engine
        self.hotkey_listener = hotkey_listener
        self.dictation = dictation
        self.speaker_model_status = (
            "ready" if embedding_engine is not None else "loading" if speaker_store else "unavailable"
        )
        self.speaker_model_error: str | None = None
        self.cleaner = None
        self.llm_engine = None
        self.cleanup_status = "loading"
        self.cleanup_error: str | None = None
        self.model_status = "ready" if transcriber is not None else "loading"
        self.model_error: str | None = None
        self.model_load_seconds: float | None = None
        self.hub = EventHub()
        self.lock = threading.Lock()
        self.session: Session | None = None
        self.awaiting_backfill: Session | None = None
        self.enrollment = None
        self._enrollment_event: dict | None = None
        self.roster: tuple[Speaker, ...] = ()
        self.meeting_name: str = "meeting"
        self.lines: list[dict] = []
        self.notes: list[dict] = []
        self.last_saved: str | None = None
        self.recovered: list[str] = []
        self.shutdown = threading.Event()

    def status(self) -> dict:
        with self.lock:
            recording = self.session is not None
            actives = (
                self.session.active
                if recording
                else {Channel.MIC: None, Channel.LOOPBACK: None}
            )
            pending_guests = (
                _guests_with_lines(self.awaiting_backfill)
                if self.awaiting_backfill is not None
                else None
            )
            speakers = []
            for index, speaker in enumerate(self.roster):
                profile_state = "untrained"
                enrollment_seconds = 0.0
                if recording and self.session is not None:
                    profile_state, enrollment_seconds = self.session.profile_status(index)
                elif speaker.speaker_id and self.speaker_store is not None:
                    try:
                        profile = self.speaker_store.profile(speaker.speaker_id)
                        profile_state = profile.state
                        enrollment_seconds = profile.enrollment_seconds
                    except KeyError:
                        pass
                spoken = bool(
                    recording
                    and self.session is not None
                    and speaker.name in self.session.log.speakers_with_lines()
                )
                speakers.append(
                    {
                        "id": speaker.speaker_id,
                        "index": index,
                        "name": speaker.name,
                        "hotkey_slot": speaker.hotkey_slot,
                        "profile_state": profile_state,
                        "enrollment_seconds": round(enrollment_seconds, 2),
                        "removed": not speaker.active,
                        "spoken": spoken,
                    }
                )
            return {
                "recording": recording,
                "speakers": speakers,
                "actives": {
                    "mic": actives[Channel.MIC],
                    "loopback": actives[Channel.LOOPBACK],
                },
                "active_speaker": (
                    self.session.current_speaker if recording else None
                ),
                "profile_learning": (
                    self.session.profile_learning if recording else None
                ),
                "pending_backfill": pending_guests,
                "meeting_name": self.meeting_name,
                "elapsed": self.session.elapsed() if recording else 0.0,
                "channels": (
                    [channel.value for channel in self.session.channels]
                    if recording
                    else []
                ),
                "backend": self.config.backend,
                "model_status": self.model_status,
                "model_error": self.model_error,
                "model_load_seconds": self.model_load_seconds,
                "settings": self.settings.to_dict(),
                "dictation": self.dictation_status(),
                "hotkey_bank": self.session.hotkey_bank if recording else 0,
                "speaker_model_status": self.speaker_model_status,
                "speaker_model_error": self.speaker_model_error,
                "cleanup_status": self.cleanup_status,
                "cleanup_error": self.cleanup_error,
                "library": (
                    self.speaker_store.library()
                    if self.speaker_store is not None
                    else {"speakers": [], "groups": [], "ready_seconds": 5.0}
                ),
                "lines": self.lines,
                "notes": self.notes,
                "last_saved": self.last_saved,
                "recovered": self.recovered,
                "enrollment": self._enrollment_event,
            }

    def model_ready(self, transcriber, elapsed: float) -> None:
        with self.lock:
            self.transcriber = transcriber
            self.model_status = "ready"
            self.model_error = None
            self.model_load_seconds = elapsed
        if self.dictation is not None:
            self.dictation.set_transcriber(transcriber)
        self.hub.publish({"type": "status", "recording": False})

    def model_failed(self, error: Exception, elapsed: float) -> None:
        with self.lock:
            self.transcriber = None
            self.model_status = "error"
            self.model_error = f"{type(error).__name__}: {error}"
            self.model_load_seconds = elapsed
        self.hub.publish({"type": "status", "recording": False})

    def speaker_model_ready(self, engine, tracking_engine=None) -> None:
        with self.lock:
            self.embedding_engine = engine
            self.tracking_embedding_engine = tracking_engine or engine
            self.speaker_model_status = "ready"
            self.speaker_model_error = None
        self.hub.publish({"type": "status", "recording": self.session is not None})

    def speaker_model_failed(self, error: Exception) -> None:
        with self.lock:
            self.embedding_engine = None
            self.tracking_embedding_engine = None
            self.speaker_model_status = "error"
            self.speaker_model_error = f"{type(error).__name__}: {error}"
        self.hub.publish({"type": "status", "recording": self.session is not None})

    def cleaner_ready(self, cleaner) -> None:
        with self.lock:
            self.cleaner = cleaner
            self.cleanup_status = "ready"
            self.cleanup_error = None
        self.hub.publish({"type": "status", "recording": self.session is not None})

    def cleaner_downloading(self) -> None:
        with self.lock:
            self.cleanup_status = "downloading"
            self.cleanup_error = None
        self.hub.publish({"type": "status", "recording": self.session is not None})

    def cleaner_unavailable(self, reason: str) -> None:
        with self.lock:
            self.cleaner = None
            self.cleanup_status = "unavailable"
            self.cleanup_error = reason
        self.hub.publish({"type": "status", "recording": self.session is not None})

    def apply_dictation_settings(self) -> None:
        """Re-bind the dictation chords from the current settings. Safe to
        call while a dictation is idle; the listener swaps bindings live."""
        if self.dictation is None:
            return
        self.dictation.rebind(dictation_options(self.settings))

    def toggle_dictation(self, body: dict) -> dict:
        """Pause or resume the dictation chord without restarting the app."""
        if self.dictation is None:
            return {"error": "dictation is unavailable"}
        wanted = body.get("enabled")
        if wanted is None:
            wanted = not self.settings.dictation_enabled
        if not isinstance(wanted, bool):
            return {"error": "enabled must be true or false"}
        return self.update_settings({"dictation_enabled": wanted})

    def dictation_history(self) -> dict:
        if self.dictation is None:
            return {"entries": [], "limit": 0}
        from .dictation.controller import HISTORY_LIMIT

        return {"entries": self.dictation.history(), "limit": HISTORY_LIMIT}

    def clear_dictation_history(self) -> dict:
        if self.dictation is None:
            return {"error": "dictation is unavailable"}
        self.dictation.clear_history()
        self.hub.publish({"type": "dictation", "phase": self.dictation.phase})
        return self.dictation_history()

    def copy_dictation(self, body: dict) -> dict:
        """Put a past dictation on the clipboard.

        Deliberately not a re-insert: the click comes from the browser, so the
        browser holds focus and the text would land there. Ctrl+Alt+Z is how
        you re-insert somewhere useful.
        """
        if self.dictation is None:
            return {"error": "dictation is unavailable"}
        entry_id = body.get("id")
        if not isinstance(entry_id, int):
            return {"error": "id must be an integer"}
        entry = next(
            (item for item in self.dictation.history() if item["id"] == entry_id), None
        )
        if entry is None:
            return {"error": "that dictation is no longer in the history"}
        try:
            from .inject import write_clipboard_text

            write_clipboard_text(entry["text"])
        except Exception as error:
            return {"error": f"could not copy: {type(error).__name__}"}
        return self.dictation_history()

    def dictation_status(self) -> dict:
        if self.dictation is None:
            return {"enabled": False, "phase": "unavailable", "error": None}
        return {
            "enabled": self.dictation.options.enabled,
            "phase": self.dictation.phase,
            "error": self.dictation.last_error,
        }

    def cleaner_failed(self, error: Exception) -> None:
        with self.lock:
            self.cleaner = None
            self.cleanup_status = "error"
            self.cleanup_error = f"{type(error).__name__}: {error}"
        self.hub.publish({"type": "status", "recording": self.session is not None})

    def update_settings(self, body: dict) -> dict:
        try:
            settings = self.settings.merged(body)
        except (TypeError, ValueError) as error:
            return {"error": str(error)}
        with self.lock:
            changes_a_chord = (
                settings.hotkey_modifiers != self.settings.hotkey_modifiers
            )
            if self.session is not None and changes_a_chord:
                return {"error": "end the meeting before changing hotkeys"}
            changes_the_model = settings.whisper_model != self.settings.whisper_model
            if self.session is not None and changes_the_model:
                return {
                    "error": "end the meeting before changing the transcription model"
                }
            try:
                if self._save_settings is not None:
                    self._save_settings(settings)
            except OSError as error:
                return {"error": f"could not save settings: {error}"}
            self.settings = settings
            if changes_the_model:
                self.config = replace(self.config, model_name=settings.whisper_model)
                self.transcriber = None
                self.model_status = "loading"
                self.model_error = None
                self.model_load_seconds = None
        if changes_the_model:
            self.reload_transcriber()
        self.apply_dictation_settings()
        self.hub.publish({"type": "status", "recording": False})
        return self.status()

    def reload_transcriber(self) -> None:
        """Rebuild the transcription model off the request thread — loading it
        takes seconds, and a settings save must not block on that."""
        thread = threading.Thread(
            target=_load_transcriber,
            args=(self, self.config),
            name="model-reloader",
            daemon=True,
        )
        thread.start()

    def _library_mutation(self, operation: Callable[[], object]) -> dict:
        with self.lock:
            if self.session is not None:
                return {"error": "end the meeting before changing the speaker library"}
            if self.speaker_store is None:
                return {"error": "speaker library is unavailable"}
            try:
                operation()
            except (KeyError, ValueError) as error:
                return {"error": str(error).strip("'")}
        self.hub.publish({"type": "status", "recording": False})
        return self.status()

    def create_library_speaker(self, body: dict) -> dict:
        return self._library_mutation(
            lambda: self.speaker_store.create_speaker(str(body.get("name", "")))
        )

    def update_library_speaker(self, body: dict) -> dict:
        return self._library_mutation(
            lambda: self.speaker_store.rename_speaker(
                str(body.get("id", "")), str(body.get("name", ""))
            )
        )

    def delete_library_speaker(self, body: dict) -> dict:
        return self._library_mutation(
            lambda: self.speaker_store.delete_speaker(str(body.get("id", "")))
        )

    def reset_library_speaker(self, body: dict) -> dict:
        return self._library_mutation(
            lambda: self.speaker_store.reset_profile(str(body.get("id", "")))
        )

    @staticmethod
    def _group_members(body: dict) -> list[dict]:
        members = body.get("members", [])
        if not isinstance(members, list):
            raise ValueError("group members must be a list")
        return members

    def create_library_group(self, body: dict) -> dict:
        return self._library_mutation(
            lambda: self.speaker_store.create_group(
                str(body.get("name", "")), self._group_members(body)
            )
        )

    def update_library_group(self, body: dict) -> dict:
        return self._library_mutation(
            lambda: self.speaker_store.update_group(
                str(body.get("id", "")),
                str(body.get("name", "")),
                self._group_members(body),
            )
        )

    def delete_library_group(self, body: dict) -> dict:
        return self._library_mutation(
            lambda: self.speaker_store.delete_group(str(body.get("id", "")))
        )

    def start_voice_recording(self, body: dict) -> dict:
        with self.lock:
            if self.session is not None:
                return {"error": "end the meeting before recording a voice sample"}
            if self.speaker_store is None or self.embedding_engine is None:
                return {"error": "speaker recognition is unavailable"}
            if self.enrollment is not None:
                return {"error": "a voice recording is already in progress"}
            speaker_id = str(body.get("id", "")).strip()
            try:
                name = self.speaker_store.profile(speaker_id).name
            except KeyError:
                return {"error": "speaker not found"}

            from .enrollment import VoiceEnrollmentRecorder

            def on_progress(progress) -> None:
                terminal = progress.phase in ("done", "stopped", "error")
                event = {
                    "type": "enrollment",
                    "speaker_id": speaker_id,
                    "name": name,
                    "phase": progress.phase,
                    "captured_seconds": round(progress.captured_seconds, 2),
                    "target_seconds": progress.target_seconds,
                    "reason": progress.reason,
                }
                self._enrollment_event = None if terminal else event
                if terminal:
                    self.enrollment = None
                self.hub.publish(event)

            self.enrollment = VoiceEnrollmentRecorder(
                self.speaker_store,
                self.embedding_engine,
                speaker_id,
                self.config,
                on_progress,
            )
            self.enrollment.start()
        return {"ok": True, "speaker_id": speaker_id}

    def stop_voice_recording(self) -> dict:
        with self.lock:
            if self.enrollment is None:
                return {"error": "no voice recording in progress"}
            self.enrollment.stop()
        return {"ok": True}

    def start_session(self, body: dict) -> dict:
        with self.lock:
            if self.session is not None:
                return {"error": "already recording"}
            if self.transcriber is None:
                if self.model_status == "error":
                    return {"error": self.model_error or "model failed to load"}
                return {"error": "transcription model is still loading"}
            from .session import Session, SessionEvents, SessionOptions

            roster = []
            for position, entry in enumerate(body.get("speakers", [])):
                if not isinstance(entry, dict):
                    continue
                speaker_id = str(entry.get("speaker_id") or "").strip() or None
                name = str(entry.get("name", "")).strip()
                if speaker_id and self.speaker_store is not None:
                    try:
                        name = self.speaker_store.profile(speaker_id).name
                    except KeyError:
                        return {"error": "saved speaker not found"}
                if not name:
                    continue
                requested_slot = entry.get("hotkey_slot", position)
                hotkey_slot = requested_slot if isinstance(requested_slot, int) else position
                roster.append(
                    Speaker(
                        name=name,
                        speaker_id=speaker_id,
                        hotkey_slot=hotkey_slot,
                    )
                )
            self.roster = tuple(roster)
            self.meeting_name = str(body.get("name") or "meeting").strip() or "meeting"
            self.lines = []
            self.notes = []
            self.last_saved = None

            def on_segment(
                segment: TranscriptSegment, label: str, latency: float, line_id: int
            ) -> None:
                session = self.session
                line = {
                    "id": line_id,
                    "time": format_timestamp(segment.start),
                    "label": label,
                    "text": segment.text,
                    "channel": segment.channel.value,
                    "speaker_id": segment.speaker_id,
                    "speaker_index": segment.speaker_index,
                    "active_speaker": (
                        session.current_speaker
                        if session is not None
                        else segment.speaker_index
                    ),
                    "attribution": segment.attribution,
                    "confidence": segment.confidence,
                }
                self.lines.append(line)
                self.hub.publish({"type": "line", **line})

            def on_line_cleaned(line_id: int, text: str | None) -> None:
                for position, line in enumerate(self.lines):
                    if line.get("id") != line_id:
                        continue
                    if text is None:
                        del self.lines[position]
                        self.hub.publish({"type": "line_drop", "id": line_id})
                    else:
                        original = line["text"]
                        line["text"] = text
                        self.hub.publish(
                            {"type": "line_update", **line, "original": original}
                        )
                    return

            def on_line_relabelled(
                line_id: int, name: str, index: int, score: float
            ) -> None:
                self._publish_line_label(line_id, name, index, "auto", score)

            def on_speaker(index: int) -> None:
                # Reads the live session roster without the state lock —
                # this fires from inside switch calls that may hold it.
                session = self.session
                if session is None or index >= len(session.roster):
                    return
                if len(session.roster) != len(self.roster):
                    self.roster = tuple(session.roster)
                    self.hub.publish({"type": "status", "recording": True})
                self.hub.publish(
                    {
                        "type": "speaker",
                        "index": index,
                        "source": "manual",
                    }
                )

            def on_attribution(
                index: int | None,
                channel: Channel,
                source: str,
                confidence: float | None,
            ) -> None:
                session = self.session
                self.hub.publish(
                    {
                        "type": "speaker",
                        "index": index,
                        "active_index": (
                            session.current_speaker if session is not None else index
                        ),
                        "channel": channel.value,
                        "source": source,
                        "confidence": confidence,
                    }
                )
                self.hub.publish({"type": "status", "recording": True})

            def on_hotkey_bank(bank: int) -> None:
                self.hub.publish({"type": "hotkey_bank", "bank": bank})

            def on_profile_learning(update) -> None:
                self.hub.publish({"type": "profile_learning", **update.to_dict()})

            def on_note(note: dict) -> None:
                self.notes.append(note)
                self.hub.publish({"type": "note", **note})

            self.awaiting_backfill = None
            self.session = Session(
                self.config,
                SessionOptions(
                    speakers=self.roster,
                    meeting_name=self.meeting_name,
                    use_loopback=bool(body.get("loopback", True)),
                    out_dir=self._out_dir,
                    hotkey_modifiers=self.settings.hotkey_modifiers,
                    hotkey_listener=self.hotkey_listener,
                    speaker_store=self.speaker_store,
                    embedding_engine=self.embedding_engine,
                    tracking_embedding_engine=self.tracking_embedding_engine,
                    cleaner=(
                        self.cleaner if bool(body.get("cleanup", True)) else None
                    ),
                ),
                self.transcriber,
                SessionEvents(
                    on_segment=on_segment,
                    on_line_cleaned=on_line_cleaned,
                    on_line_relabelled=on_line_relabelled,
                    on_speaker=on_speaker,
                    on_attribution=on_attribution,
                    on_hotkey_bank=on_hotkey_bank,
                    on_profile_learning=on_profile_learning,
                    on_note=on_note,
                ),
            )
            self.session.start()
        self.hub.publish({"type": "status", "recording": True})
        return self.status()

    def stop_session(self, discard: bool = False) -> dict:
        with self.lock:
            if self.session is None:
                return {"error": "not recording"}
            session = self.session
            self.session = None
        session.stop()
        needs_backfill = (
            not discard
            and bool(_guests_with_lines(session))
            and not session.log.is_empty
        )
        with self.lock:
            if needs_backfill:
                self.awaiting_backfill = session
            elif discard:
                session.discard()
                self.last_saved = None
            else:
                saved = session.save()
                self.last_saved = str(saved) if saved else None
        self.hub.publish({"type": "status", "recording": False})
        return self.status()

    def finalize(self, body: dict) -> dict:
        with self.lock:
            if self.awaiting_backfill is None:
                return {"error": "nothing awaiting backfill"}
            session = self.awaiting_backfill
            renames = {
                str(old): str(new)
                for old, new in (body.get("renames") or {}).items()
                if str(new).strip()
            }
            guest_profiles = body.get("guest_profiles") or {}
            if not isinstance(guest_profiles, dict):
                return {"error": "guest_profiles must be an object"}
            if self.speaker_store is not None:
                try:
                    for guest_name, selection in guest_profiles.items():
                        if not isinstance(selection, dict):
                            continue
                        speaker_id = str(selection.get("speaker_id") or "").strip()
                        if speaker_id:
                            profile = self.speaker_store.profile(speaker_id)
                        else:
                            new_name = str(selection.get("name") or "").strip()
                            if not new_name:
                                continue
                            profile = self.speaker_store.create_speaker(new_name)
                        session.persist_guest(str(guest_name), profile.speaker_id)
                        renames[str(guest_name)] = profile.name
                except (KeyError, ValueError) as error:
                    return {"error": str(error).strip("'")}
            self.awaiting_backfill = None
            saved = session.save(renames)
            self.last_saved = str(saved) if saved else None
            for line in self.lines:
                if line["label"] in renames:
                    line["label"] = renames[line["label"]]
        self.hub.publish({"type": "status", "recording": False})
        return self.status()

    def add_guest(self, body: dict) -> dict:
        with self.lock:
            session = self.session
            if session is None:
                return {"error": "not recording"}
            session.add_guest()
            self.roster = tuple(session.roster)
        self.hub.publish({"type": "status", "recording": True})
        return self.status()

    def add_roster_speaker(self, body: dict) -> dict:
        with self.lock:
            session = self.session
            if session is None:
                return {"error": "not recording"}
            speaker_id = str(body.get("speaker_id") or "").strip() or None
            name = str(body.get("name", "")).strip()
            if speaker_id and self.speaker_store is not None:
                try:
                    name = self.speaker_store.profile(speaker_id).name
                except KeyError:
                    return {"error": "speaker not found"}
            if not name:
                return {"error": "a name is required"}
            session.add_speaker(name, speaker_id)
            self.roster = tuple(session.roster)
        self.hub.publish({"type": "status", "recording": True})
        return self.status()

    def remove_roster_speaker(self, body: dict) -> dict:
        with self.lock:
            session = self.session
            if session is None:
                return {"error": "not recording"}
            index = body.get("index")
            if not isinstance(index, int):
                return {"error": "index must be an integer"}
            error = session.remove_speaker(index)
            if error:
                return {"error": error}
            self.roster = tuple(session.roster)
        self.hub.publish({"type": "status", "recording": True})
        return self.status()

    def rename_roster_speaker(self, body: dict) -> dict:
        with self.lock:
            session = self.session
            if session is None:
                return {"error": "not recording"}
            index = body.get("index")
            if not isinstance(index, int):
                return {"error": "index must be an integer"}
            old = (
                session.roster[index].name
                if 0 <= index < len(session.roster)
                else None
            )
            speaker_id = str(body.get("speaker_id") or "").strip() or None
            if speaker_id is not None and self.speaker_store is not None:
                try:
                    self.speaker_store.profile(speaker_id)
                except KeyError:
                    return {"error": "speaker not found"}
            error = session.rename_speaker(
                index, str(body.get("name", "")), speaker_id
            )
            if error:
                return {"error": error}
            new_name = session.roster[index].name
            if old is not None and old != new_name:
                for line in self.lines:
                    if line.get("label") == old:
                        line["label"] = new_name
            self.roster = tuple(session.roster)
        self.hub.publish({"type": "status", "recording": True})
        return self.status()

    def _publish_line_label(
        self,
        line_id: int,
        name: str,
        index: int | None,
        source: str,
        confidence: float | None,
    ) -> None:
        """Re-render one transcript line under a new speaker. Rides the same
        line_update event the cleanup pass already uses, so the browser needs
        nothing new to show it."""
        for line in self.lines:
            if line.get("id") != line_id:
                continue
            line["label"] = name
            line["speaker_index"] = index
            line["attribution"] = source
            line["confidence"] = confidence
            self.hub.publish({"type": "line_update", **line})
            return

    def assign_line(self, body: dict) -> dict:
        """Put a transcript line on a person by hand.

        Recording only: once the meeting stops the session — and with it the
        log this writes through to — is gone.
        """
        with self.lock:
            session = self.session
            if session is None:
                return {"error": "not recording"}
            line_id = body.get("id")
            index = body.get("index")
            if not isinstance(line_id, int):
                return {"error": "id must be an integer"}
            # bool is an int subclass, so it has to be excluded explicitly.
            if index is not None and (
                isinstance(index, bool) or not isinstance(index, int)
            ):
                return {"error": "index must be an integer or null"}
            error = session.assign_line(line_id, index)
            if error:
                return {"error": error}
            name = "Unknown" if index is None else session.roster[index].name
            self._publish_line_label(line_id, name, index, "manual", None)
        return self.status()

    def reset_roster_profile(self, body: dict) -> dict:
        """The in-meeting counterpart to /api/library/speakers/reset, which
        _library_mutation refuses while a session is live."""
        with self.lock:
            session = self.session
            if session is None:
                return {"error": "not recording"}
            index = body.get("index")
            if not isinstance(index, int):
                return {"error": "index must be an integer"}
            error = session.reset_speaker_profile(index)
            if error:
                return {"error": error}
        self.hub.publish({"type": "status", "recording": True})
        return self.status()

    def cancel_profile_learning(self) -> dict:
        with self.lock:
            session = self.session
        if session is None:
            return {"error": "not recording"}
        session.cancel_profile_learning()
        return self.status()

    def add_note(self, body: dict) -> dict:
        with self.lock:
            session = self.session
            if session is None:
                return {"error": "not recording"}
            text = str(body.get("text", "")).strip()
            if not text:
                return {"error": "note text is required"}
            session.add_note(text)
        return self.status()

    def switch_speaker(self, index: int) -> dict:
        with self.lock:
            session = self.session
        if session is not None:
            session.switch_speaker(index)
        return self.status()

    def page_hotkeys(self, direction: int) -> dict:
        with self.lock:
            session = self.session
        if session is not None:
            session.page_hotkeys(direction)
        return self.status()


def is_trusted_request(host_header: str | None, origin_header: str | None, port: int) -> bool:
    """Blocks DNS-rebinding (Host must name this loopback port) and
    cross-site POSTs (a present Origin must match too)."""
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if host_header not in allowed_hosts:
        return False
    if origin_header is None:
        return True
    allowed_origins = {f"http://{host}" for host in allowed_hosts}
    return origin_header in allowed_origins


def _guests_with_lines(session: Session) -> list[str]:
    spoke = session.log.speakers_with_lines()
    return [
        session.roster[index].name
        for index in session.guest_indices
        if session.roster[index].name in spoke
    ]


def _load_asset(name: str) -> bytes:
    return (resources.files("oat_notes") / "web" / name).read_bytes()


class Handler(BaseHTTPRequestHandler):
    state: AppState  # assigned by serve()
    port: int  # assigned by serve()

    def log_message(self, format: str, *log_args) -> None:
        pass  # keep the console clean; transcript lines matter more

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus | None = None) -> None:
        if status is None:
            status = HTTPStatus.CONFLICT if "error" in payload else HTTPStatus.OK
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _is_trusted(self) -> bool:
        return is_trusted_request(
            self.headers.get("Host"), self.headers.get("Origin"), self.port
        )

    def do_GET(self) -> None:
        if not self._is_trusted():
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            return
        if self.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, _load_asset("index.html"), "text/html; charset=utf-8")
        elif self.path == "/favicon.svg":
            self._send(HTTPStatus.OK, _load_asset("favicon.svg"), "image/svg+xml")
        elif self.path == "/pixel.woff2":
            self._send(HTTPStatus.OK, _load_asset("pixel.woff2"), "font/woff2")
        elif self.path == "/api/state":
            self._send_json(self.state.status())
        elif self.path == "/api/dictation/history":
            self._send_json(self.state.dictation_history())
        elif self.path == "/api/library":
            self._send_json(
                self.state.speaker_store.library()
                if self.state.speaker_store is not None
                else {"speakers": [], "groups": [], "ready_seconds": 5.0}
            )
        elif self.path == "/api/events":
            self._serve_events()
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:
        if not self._is_trusted():
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
        except ValueError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/start":
            self._send_json(self.state.start_session(body))
        elif self.path == "/api/stop":
            self._send_json(self.state.stop_session(discard=bool(body.get("discard"))))
        elif self.path == "/api/switch":
            index = body.get("index")
            if not isinstance(index, int):
                self._send_json(
                    {"error": "index must be an integer"}, status=HTTPStatus.BAD_REQUEST
                )
                return
            self._send_json(self.state.switch_speaker(index))
        elif self.path == "/api/dictation/toggle":
            self._send_json(self.state.toggle_dictation(body))
        elif self.path == "/api/dictation/history/clear":
            self._send_json(self.state.clear_dictation_history())
        elif self.path == "/api/dictation/copy":
            self._send_json(self.state.copy_dictation(body))
        elif self.path == "/api/add_guest":
            self._send_json(self.state.add_guest(body))
        elif self.path == "/api/roster/add":
            self._send_json(self.state.add_roster_speaker(body))
        elif self.path == "/api/roster/remove":
            self._send_json(self.state.remove_roster_speaker(body))
        elif self.path == "/api/roster/rename":
            self._send_json(self.state.rename_roster_speaker(body))
        elif self.path == "/api/roster/reset_profile":
            self._send_json(self.state.reset_roster_profile(body))
        elif self.path == "/api/lines/assign":
            self._send_json(self.state.assign_line(body))
        elif self.path == "/api/profile_learning/cancel":
            self._send_json(self.state.cancel_profile_learning())
        elif self.path == "/api/notes/add":
            self._send_json(self.state.add_note(body))
        elif self.path == "/api/hotkey_bank":
            direction = body.get("direction")
            if direction not in (-1, 1):
                self._send_json(
                    {"error": "direction must be -1 or 1"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json(self.state.page_hotkeys(direction))
        elif self.path == "/api/finalize":
            self._send_json(self.state.finalize(body))
        elif self.path == "/api/settings":
            self._send_json(self.state.update_settings(body))
        elif self.path == "/api/library/speakers/create":
            self._send_json(self.state.create_library_speaker(body))
        elif self.path == "/api/library/speakers/update":
            self._send_json(self.state.update_library_speaker(body))
        elif self.path == "/api/library/speakers/delete":
            self._send_json(self.state.delete_library_speaker(body))
        elif self.path == "/api/library/speakers/reset":
            self._send_json(self.state.reset_library_speaker(body))
        elif self.path == "/api/library/groups/create":
            self._send_json(self.state.create_library_group(body))
        elif self.path == "/api/library/groups/update":
            self._send_json(self.state.update_library_group(body))
        elif self.path == "/api/library/groups/delete":
            self._send_json(self.state.delete_library_group(body))
        elif self.path == "/api/library/speakers/record/start":
            self._send_json(self.state.start_voice_recording(body))
        elif self.path == "/api/library/speakers/record/stop":
            self._send_json(self.state.stop_voice_recording())
        elif self.path == "/api/quit":
            self._send_json({"ok": True})
            self.state.shutdown.set()
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def _serve_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        subscriber = self.state.hub.subscribe()
        try:
            while not self.state.shutdown.is_set():
                try:
                    event = subscriber.get(timeout=15.0)
                    payload = f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    payload = ": keepalive\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
        except (ConnectionAbortedError, ConnectionResetError, OSError):
            pass
        finally:
            self.state.hub.unsubscribe(subscriber)


def _load_transcriber(state: AppState, config: Config) -> None:
    """Build and warm the transcription model, publishing the outcome. Called
    once at startup and again whenever Settings picks a different model."""
    started = time.perf_counter()
    try:
        from .cli import warm_up
        from .transcriber import create_transcriber

        print(f"Loading transcription model ({config.model_name})…", flush=True)
        transcriber = create_transcriber(config)
        warm_up(config, transcriber)
    except Exception as error:
        elapsed = time.perf_counter() - started
        print(f"Model initialization failed after {elapsed:.1f}s: {error}", file=sys.stderr)
        state.model_failed(error, elapsed)
    else:
        elapsed = time.perf_counter() - started
        print(f"Transcription model ready in {elapsed:.1f}s", flush=True)
        state.model_ready(transcriber, elapsed)


def _initialize_model(state: AppState, config: Config) -> None:
    """Load both local inference engines off the UI thread."""
    _load_transcriber(state, config)

    try:
        from .speaker_id import SherpaOnnxEmbeddingEngine

        engine = SherpaOnnxEmbeddingEngine()
    except Exception as error:
        print(
            f"Speaker recognition unavailable: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        state.speaker_model_failed(error)
    else:
        # A second extractor instance lets live speaker tracking run
        # without queuing behind per-turn attribution on the same model
        # session; fall back to sharing one instance if it won't load.
        tracking_engine = engine
        try:
            tracking_engine = SherpaOnnxEmbeddingEngine()
        except Exception as error:
            print(
                "Live speaker tracking will share the main speaker model "
                f"(second instance failed: {type(error).__name__}: {error})",
                file=sys.stderr,
            )
        print("Local speaker recognition ready", flush=True)
        state.speaker_model_ready(engine, tracking_engine)

    if not config.cleanup_enabled:
        state.cleaner_unavailable("disabled")
        return
    try:
        from .cleanup import TranscriptCleaner
        from .llm import LlmEngine, model_is_cached

        if not config.offline and not model_is_cached(config):
            print(f"Downloading language model ({config.cleanup_model})…", flush=True)
            state.cleaner_downloading()
        else:
            print(f"Loading language model ({config.cleanup_model})…", flush=True)
        engine = LlmEngine.from_config(config)
        cleaner = TranscriptCleaner(engine)
        cleaner.warm_up()
    except ImportError:
        print(
            "Transcript cleanup unavailable: install the openvino extra"
            " (uv sync --extra openvino)",
            file=sys.stderr,
        )
        state.cleaner_unavailable("openvino extra not installed")
    except Exception as error:
        print(
            f"Transcript cleanup unavailable: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        state.cleaner_failed(error)
    else:
        print(f"Language model ready on {engine.device}", flush=True)
        state.llm_engine = engine
        if state.dictation is not None:
            state.dictation.set_engine(engine)
        state.cleaner_ready(cleaner)


def exit_backstop(seconds: float = 5.0) -> threading.Timer:
    """Guarantee the process actually dies after an orderly shutdown.

    Teardown stops the captures, drains the pipeline and closes the server, but
    the native audio and inference libraries underneath can leave a thread that
    outlives the interpreter. A process that lingers keeps its low-level
    keyboard hook installed, so the next launch runs two — and the stale one
    still answers on its port until it finally releases it.

    Nothing is masked: this fires only if the normal exit has not happened,
    and stderr is flushed first because a windowed build's stderr is the log.
    """
    def bail() -> None:
        sys.stderr.flush()
        os._exit(0)

    timer = threading.Timer(seconds, bail)
    timer.name = "exit-backstop"
    timer.daemon = True
    timer.start()
    return timer


def _await_shutdown(state: AppState, overlay, show_overlay: bool = True) -> None:
    """Block the main thread until the app is told to quit.

    Tk insists on owning the thread it was created on, so when the overlay is
    enabled the main thread becomes its event loop and polls ``shutdown``
    from inside. Without it — headless runs, or a build with no tkinter —
    this stays the plain wait it always was.
    """
    if show_overlay:
        try:
            overlay.run(state.shutdown)
            return
        except Exception as error:
            print(
                f"overlay unavailable, continuing without it: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
    while not state.shutdown.wait(timeout=0.3):
        pass


def serve(config: Config, args: argparse.Namespace, open_browser: bool = True) -> None:
    started = time.perf_counter()
    settings_store = SettingsStore()
    speaker_store = SpeakerStore()
    settings = settings_store.load()
    # An explicit --model is the operator's choice for this run and outranks
    # the saved preference; without one, Settings decides.
    if getattr(args, "model", None) is None:
        config = replace(config, model_name=settings.whisper_model)
    hotkey_listener = HotkeyListener()
    url = f"http://127.0.0.1:{args.port}"
    overlay = Overlay(
        on_quit=lambda: state.shutdown.set(),
        on_open_ui=lambda: webbrowser.open(url),
    )
    last_phase_sent_to_browser: list[str] = [""]

    def on_dictation_phase(phase: str, details: dict) -> None:
        overlay.post_phase(phase, details)
        if last_phase_sent_to_browser[0] != phase:
            last_phase_sent_to_browser[0] = phase
            state.hub.publish({"type": "dictation", "phase": phase, **details})

    dictation = DictationController(
        config,
        options=dictation_options(settings),
        on_phase=on_dictation_phase,
        level_sink=overlay.post_level,
    )
    state = AppState(
        config,
        transcriber=None,
        out_dir=args.out_dir,
        settings=settings,
        save_settings=settings_store.save,
        speaker_store=speaker_store,
        hotkey_listener=hotkey_listener,
        dictation=dictation,
    )
    recovered = recover_journals(args.out_dir)
    if recovered:
        state.recovered = [str(path) for path in recovered]
        for path in recovered:
            print(f"Recovered an unsaved transcript from a previous run: {path}", flush=True)
    Handler.state = state
    Handler.port = args.port
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        if getattr(error, "winerror", None) == 10048 or error.errno == 98:
            print(f"oat-notes is already running at {url}", flush=True)
            if open_browser:
                webbrowser.open(url)
            return
        raise
    server.daemon_threads = True

    dictation.bind(hotkey_listener)
    dictation.start()
    hotkey_listener.start()

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(
        f"oat-notes UI ready in {time.perf_counter() - started:.2f}s: {url}",
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    # OpenVINO construction can hold the interpreter lock briefly. Give the
    # browser enough time to fetch and paint the tiny UI before it begins.
    model_thread = threading.Timer(
        0.75,
        _initialize_model,
        args=(state, config),
    )
    model_thread.name = "model-loader"
    model_thread.daemon = True
    model_thread.start()
    try:
        _await_shutdown(
            state,
            overlay,
            show_overlay=settings.overlay_enabled and not args.no_overlay,
        )
    except KeyboardInterrupt:
        pass
    print("Shutting down…", flush=True)
    exit_backstop()
    dictation.stop()
    hotkey_listener.stop()
    if state.session is not None:
        state.stop_session()
    server.shutdown()
