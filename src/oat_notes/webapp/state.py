"""Everything the request handlers share: one meeting at a time, the model
readiness flags, and the status report the browser polls."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..attribution import Speaker
from ..config import Config
from ..dictation.controller import DictationController, DictationOptions
from ..hotkeys import HotkeyListener
from ..line_feed import LineFeed
from ..settings import AppSettings
from ..speaker_id import SpeakerEmbeddingEngine
from ..speaker_store import READY_SPEECH_SECONDS, SpeakerStore
from ..transcriber import Transcriber
from ..types import Channel
from .hub import EventHub
from .requests import conflict

if TYPE_CHECKING:
    from ..cleanup import LineCleaner
    from ..enrollment import VoiceEnrollmentRecorder
    from ..session import Session

DEFAULT_SETTINGS = AppSettings()
EMPTY_LIBRARY = {"speakers": [], "groups": [], "ready_seconds": READY_SPEECH_SECONDS}


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


def guests_with_lines(session: Session) -> list[str]:
    spoke = session.log.speakers_with_lines()
    return [
        session.roster[index].name
        for index in session.guest_indices
        if session.roster[index].name in spoke
    ]


class AppState:
    def __init__(
        self,
        config: Config,
        transcriber: Transcriber | None,
        out_dir: Path = Path("transcripts"),
        settings: AppSettings = DEFAULT_SETTINGS,
        save_settings: Callable[[AppSettings], None] | None = None,
        speaker_store: SpeakerStore | None = None,
        embedding_engine: SpeakerEmbeddingEngine | None = None,
        tracking_embedding_engine: SpeakerEmbeddingEngine | None = None,
        hotkey_listener: HotkeyListener | None = None,
        dictation: DictationController | None = None,
        shutdown: threading.Event | None = None,
        hub: EventHub | None = None,
    ) -> None:
        self.config = config
        self.transcriber = transcriber
        self.out_dir = out_dir
        self.settings = settings
        self.save_settings = save_settings
        self.speaker_store = speaker_store
        self.embedding_engine = embedding_engine
        self.tracking_embedding_engine = tracking_embedding_engine
        self.hotkey_listener = hotkey_listener
        self.dictation = dictation
        self.speaker_model_status = (
            "ready" if embedding_engine is not None else "loading" if speaker_store else "unavailable"
        )
        self.speaker_model_error: str | None = None
        self.cleaner: LineCleaner | None = None
        self.cleanup_status = "loading"
        self.cleanup_error: str | None = None
        self.model_status = "ready" if transcriber is not None else "loading"
        self.model_error: str | None = None
        self.model_load_seconds: float | None = None
        self.hub = hub or EventHub()
        self.lock = threading.Lock()
        self.session: Session | None = None
        self.awaiting_backfill: Session | None = None
        self.enrollment: VoiceEnrollmentRecorder | None = None
        self.enrollment_event: dict | None = None
        self.roster: tuple[Speaker, ...] = ()
        self.meeting_name: str = "meeting"
        self.lines = LineFeed()
        self.last_saved: str | None = None
        self.recovered: list[str] = []
        self.shutdown = shutdown or threading.Event()

    def publish_status(self) -> None:
        self.hub.publish({"type": "status", "recording": self.session is not None})

    def require_session(self) -> Session:
        """Caller holds ``lock``."""
        if self.session is None:
            raise conflict("not recording")
        return self.session

    def sync_roster(self, session: Session) -> None:
        self.roster = tuple(session.roster)

    def library(self) -> dict:
        if self.speaker_store is None:
            return dict(EMPTY_LIBRARY)
        return self.speaker_store.library()

    def status(self) -> dict:
        lines, notes = self.lines.snapshot()
        with self.lock:
            session = self.session
            recording = session is not None
            report = {
                "recording": recording,
                "speakers": [
                    self._speaker_status(index, speaker, session)
                    for index, speaker in enumerate(self.roster)
                ],
                "pending_backfill": (
                    guests_with_lines(self.awaiting_backfill)
                    if self.awaiting_backfill is not None
                    else None
                ),
                "meeting_name": self.meeting_name,
                "backend": self.config.backend,
                "model_status": self.model_status,
                "model_error": self.model_error,
                "model_load_seconds": self.model_load_seconds,
                "settings": self.settings.to_dict(),
                "dictation": self.dictation_status(),
                "speaker_model_status": self.speaker_model_status,
                "speaker_model_error": self.speaker_model_error,
                "cleanup_status": self.cleanup_status,
                "cleanup_error": self.cleanup_error,
                "library": self.library(),
                "lines": lines,
                "notes": notes,
                "last_saved": self.last_saved,
                "recovered": self.recovered,
                "enrollment": self.enrollment_event,
            }
            report.update(self._session_status(session))
            return report

    def _speaker_status(self, index: int, speaker: Speaker, session: Session | None) -> dict:
        profile_state, enrollment_seconds = "untrained", 0.0
        if session is not None:
            profile_state, enrollment_seconds = session.profile_status(index)
        elif speaker.speaker_id and self.speaker_store is not None:
            profile = self.speaker_store.find(speaker.speaker_id)
            if profile is not None:
                profile_state, enrollment_seconds = profile.state, profile.enrollment_seconds
        return {
            "id": speaker.speaker_id,
            "index": index,
            "name": speaker.name,
            "hotkey_slot": speaker.hotkey_slot,
            "profile_state": profile_state,
            "enrollment_seconds": round(enrollment_seconds, 2),
            "removed": not speaker.active,
            "spoken": session is not None and speaker.name in session.log.speakers_with_lines(),
        }

    @staticmethod
    def _session_status(session: Session | None) -> dict:
        if session is None:
            return {
                "actives": {"mic": None, "loopback": None},
                "active_speaker": None,
                "profile_learning": None,
                "elapsed": 0.0,
                "channels": [],
                "hotkey_bank": 0,
            }
        return {
            "actives": {
                "mic": session.active[Channel.MIC],
                "loopback": session.active[Channel.LOOPBACK],
            },
            "active_speaker": session.current_speaker,
            "profile_learning": session.profile_learning,
            "elapsed": session.elapsed(),
            "channels": [channel.value for channel in session.channels],
            "hotkey_bank": session.hotkey_bank,
        }

    def dictation_status(self) -> dict:
        if self.dictation is None:
            return {"enabled": False, "phase": "unavailable", "error": None}
        return {
            "enabled": self.dictation.options.enabled,
            "phase": self.dictation.phase,
            "error": self.dictation.last_error,
        }

    def apply_dictation_settings(self) -> None:
        """Re-bind the dictation chords from the current settings. Safe to
        call while a dictation is idle; the listener swaps bindings live."""
        if self.dictation is not None:
            self.dictation.rebind(dictation_options(self.settings))

    def model_ready(self, transcriber: Transcriber, elapsed: float) -> None:
        with self.lock:
            self.transcriber = transcriber
            self.model_status = "ready"
            self.model_error = None
            self.model_load_seconds = elapsed
        if self.dictation is not None:
            self.dictation.set_transcriber(transcriber)
        self.publish_status()

    def model_failed(self, error: Exception, elapsed: float) -> None:
        with self.lock:
            self.transcriber = None
            self.model_status = "error"
            self.model_error = f"{type(error).__name__}: {error}"
            self.model_load_seconds = elapsed
        self.publish_status()

    def speaker_model_ready(
        self,
        engine: SpeakerEmbeddingEngine,
        tracking_engine: SpeakerEmbeddingEngine | None = None,
    ) -> None:
        with self.lock:
            self.embedding_engine = engine
            self.tracking_embedding_engine = tracking_engine or engine
            self.speaker_model_status = "ready"
            self.speaker_model_error = None
        self.publish_status()

    def speaker_model_failed(self, error: Exception) -> None:
        with self.lock:
            self.embedding_engine = None
            self.tracking_embedding_engine = None
            self.speaker_model_status = "error"
            self.speaker_model_error = f"{type(error).__name__}: {error}"
        self.publish_status()

    def cleaner_ready(self, cleaner: LineCleaner) -> None:
        with self.lock:
            self.cleaner = cleaner
            self.cleanup_status = "ready"
            self.cleanup_error = None
        self.publish_status()

    def cleaner_downloading(self) -> None:
        with self.lock:
            self.cleanup_status = "downloading"
            self.cleanup_error = None
        self.publish_status()

    def cleaner_unavailable(self, reason: str) -> None:
        with self.lock:
            self.cleaner = None
            self.cleanup_status = "unavailable"
            self.cleanup_error = reason
        self.publish_status()

    def cleaner_failed(self, error: Exception) -> None:
        with self.lock:
            self.cleaner = None
            self.cleanup_status = "error"
            self.cleanup_error = f"{type(error).__name__}: {error}"
        self.publish_status()
