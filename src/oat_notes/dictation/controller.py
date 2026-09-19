"""The dictation state machine: chord -> capture -> format -> insert.

Hotkey callbacks arrive on the low-level keyboard hook thread, so they only
drop a command on a queue; one worker thread does the real work. That also
serializes it — opening the mic, transcribing and inserting can never
interleave with the next chord press.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from .. import inject
from ..config import Config
from ..hotkeys import Chord, HotkeyListener
from ..log import error_kind
from ..settings import REPLAY_KEY
from ..transcriber import Transcriber
from .format import format_dictation, vocabulary_hotwords
from .phases import (
    CANCELLED,
    ERROR,
    FORMATTING,
    IDLE,
    INSERTED,
    LATCHED,
    LISTENING,
    LOADING,
    TRANSCRIBING,
    DictationPhase,
)
from .recorder import UtteranceRecorder

log = logging.getLogger(__name__)

_ARM: Final = "arm"
_CANCEL: Final = "cancel"
_RELEASE: Final = "release"
_REPLAY: Final = "replay"
_SHUTDOWN: Final = "shutdown"
_CommandKind = Literal["arm", "cancel", "release", "replay", "shutdown"]

_LATCH_SPEECH_CEILING = 0.35
HISTORY_LIMIT = 20


@dataclass(frozen=True)
class DictationOptions:
    enabled: bool = True
    modifiers: tuple[str, ...] = ("ctrl", "win")
    email_modifiers: tuple[str, ...] = ("ctrl", "shift", "win")
    replay_modifiers: tuple[str, ...] = ("ctrl", "alt")
    tap_seconds: float = 0.4
    restore_clipboard: bool = True
    device_index: int | None = None
    email_detection: bool = True
    spoken_punctuation: bool = False
    vocabulary: tuple[tuple[str, str], ...] = ()


@dataclass
class _Command:
    kind: _CommandKind
    hwnd: int = 0
    held_seconds: float = 0.0
    force_email: bool = False


class DictationController:
    """Owns the recorder and the chord bindings. ``format_text`` maps the raw
    transcript to the text to insert plus a mode label."""

    def __init__(
        self,
        config: Config,
        options: DictationOptions | None = None,
        on_phase: Callable[[DictationPhase, dict], None] | None = None,
        level_sink: Callable[[float], None] | None = None,
        format_text: Callable[[str, bool], tuple[str, str]] | None = None,
    ) -> None:
        self._config = config
        self.options = options or DictationOptions()
        self._on_phase = on_phase
        self._level_sink = level_sink
        self._format_text = format_text or self._default_format
        self._engine = None

        self._transcriber: Transcriber | None = None
        self._recorder: UtteranceRecorder | None = None
        self._commands: queue.Queue[_Command] = queue.Queue(maxsize=32)
        self._worker: threading.Thread | None = None
        self._listener: HotkeyListener | None = None

        self._recording = False
        self._latched = False
        self._force_email = False
        self._target_hwnd = 0
        self.last_text: str | None = None
        # Deliberately memory-only: this is a verbatim record of everything
        # dictated, and the application never writes captured text to disk.
        self._history: deque[dict] = deque(maxlen=HISTORY_LIMIT)
        self._history_lock = threading.Lock()
        self._next_entry_id = 1

        self.phase = IDLE
        self.last_error: str | None = None

    def _default_format(self, raw: str, force_email: bool) -> tuple[str, str]:
        """Rules always; the LLM only for email, and only once it has loaded,
        so dictation stays usable while the language model is warming up."""
        return format_dictation(
            raw,
            engine=self._engine,
            force_email=force_email,
            vocabulary=self.options.vocabulary,
            spoken_punctuation=self.options.spoken_punctuation,
            detect=self.options.email_detection,
        )

    def set_engine(self, engine) -> None:
        self._engine = engine

    def set_transcriber(self, transcriber: Transcriber) -> None:
        """Called from the model loader. Building the recorder here opens
        PyAudio and the Silero session on that background thread rather than
        on the first hotkey press."""
        self._transcriber = transcriber
        self._recorder = UtteranceRecorder(
            self._config,
            transcriber,
            device_index=self.options.device_index,
            level_sink=self._level_sink,
            hotwords=vocabulary_hotwords(self.options.vocabulary),
        )
        try:
            self._recorder.prepare()
        except Exception as error:
            log.error("dictation audio unavailable: %s: %s", error_kind(error), error)
            self._recorder = None
            self._publish(ERROR, error=str(error))
            return
        self._publish(IDLE)

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._run, name="dictation-controller", daemon=True
        )
        self._worker.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._commands.put(_Command(_SHUTDOWN))
            self._worker.join(timeout=5.0)
            self._worker = None
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    def bind(self, listener: HotkeyListener) -> None:
        self._listener = listener
        if not self.options.enabled:
            return
        listener.bind(
            "dictation",
            Chord.of(self.options.modifiers),
            on_press=lambda: self._post(_ARM, hwnd=self._capture_target()),
            on_release=lambda held: self._post(_RELEASE, held_seconds=held),
            on_cancel=lambda: self._post(_CANCEL),
        )
        if self.options.email_modifiers:
            listener.bind(
                "dictation-email",
                Chord.of(self.options.email_modifiers),
                on_press=lambda: self._post(
                    _ARM, hwnd=self._capture_target(), force_email=True
                ),
                on_release=lambda held: self._post(_RELEASE, held_seconds=held),
                on_cancel=lambda: self._post(_CANCEL),
            )
        if self.options.replay_modifiers:
            listener.bind(
                "dictation-replay",
                Chord.of(self.options.replay_modifiers, REPLAY_KEY),
                on_press=lambda: self._post(_REPLAY, hwnd=self._capture_target()),
            )
        listener.bind("dictation-cancel", Chord.of((), "esc"), self._cancel_if_active)

    def unbind(self, listener: HotkeyListener) -> None:
        for name in (
            "dictation", "dictation-email", "dictation-replay", "dictation-cancel"
        ):
            listener.unbind(name)

    def rebind(self, options: DictationOptions) -> None:
        self._post(_CANCEL)
        self.options = options
        if self._recorder is not None:
            self._recorder.hotwords = vocabulary_hotwords(options.vocabulary)
        if self._listener is not None:
            self.unbind(self._listener)
            self.bind(self._listener)

    def _cancel_if_active(self) -> None:
        if self._recording:
            self._post(_CANCEL)

    @staticmethod
    def _capture_target() -> int:
        try:
            return inject.foreground_window()
        except Exception as error:
            log.warning("could not read the foreground window: %s", error_kind(error))
            return 0

    def _post(self, kind: _CommandKind, **fields) -> bool:
        try:
            self._commands.put_nowait(_Command(kind, **fields))
        except queue.Full:
            log.warning("dictation command %s dropped: worker queue is full", kind)
            return False
        return True

    def _run(self) -> None:
        while True:
            command = self._commands.get()
            if command.kind == _SHUTDOWN:
                if self._recording:
                    self._abandon()
                return
            self._dispatch(command)

    def _dispatch(self, command: _Command) -> None:
        """Keeps the worker alive and the mic released when anything
        downstream throws."""
        try:
            self._handle(command)
        except Exception as error:
            log.error("dictation failed: %s", error_kind(error))
            self._abandon()
            self._publish(ERROR, error=str(error))

    def _handle(self, command: _Command) -> None:
        if command.kind == _ARM:
            self._on_arm(command)
        elif command.kind == _CANCEL:
            self._on_cancel()
        elif command.kind == _RELEASE:
            self._on_release(command)
        elif command.kind == _REPLAY:
            self._on_replay(command)

    def _on_replay(self, command: _Command) -> None:
        """Re-insert the last dictation wherever the cursor is now, which is
        the point: the usual reason to replay is that it landed in the wrong
        window the first time."""
        if self._recording or not self.last_text:
            return
        inject.inject(
            self.last_text,
            hwnd=command.hwnd,
            restore_clipboard=self.options.restore_clipboard,
        )
        self._publish(INSERTED, mode="replay", characters=len(self.last_text))

    def _on_arm(self, command: _Command) -> None:
        if self._recording:
            if self._latched:
                self._commit()
            return
        if self._recorder is None:
            self._publish(LOADING)
            return
        self._target_hwnd = command.hwnd
        self._force_email = command.force_email
        self._recorder.start()
        self._recording = True
        self._latched = False
        self._publish(LISTENING, email=self._force_email)

    def _on_cancel(self) -> None:
        if not self._recording:
            return
        self._abandon()
        self._publish(CANCELLED)

    def _on_release(self, command: _Command) -> None:
        if not self._recording or self._latched:
            return
        if self._should_latch(command.held_seconds):
            self._latched = True
            self._publish(LATCHED, email=self._force_email)
            return
        self._commit()

    def _should_latch(self, held_seconds: float) -> bool:
        """A quick tap that caught no real speech means hands-free, not a
        half-second dictation."""
        if held_seconds >= self.options.tap_seconds:
            return False
        recorder = self._recorder
        return recorder is None or recorder.speech_seconds <= _LATCH_SPEECH_CEILING

    def _commit(self) -> None:
        self._publish(TRANSCRIBING)
        raw = self._recorder.stop()
        self._recording = False
        self._latched = False
        if not raw:
            self._publish(IDLE)
            return
        self._publish(FORMATTING, email=self._force_email)
        text, mode = self._format_text(raw, self._force_email)
        if not text:
            self._publish(IDLE)
            return
        # Recorded before the insert, not after: if the target window is gone
        # the insert raises, and the transcript has to survive that or the
        # dictation is lost for good.
        self.last_text = text
        self._remember(text, mode)
        inject.inject(
            text,
            hwnd=self._target_hwnd,
            restore_clipboard=self.options.restore_clipboard,
        )
        self._publish(INSERTED, mode=mode, characters=len(text))

    def _remember(self, text: str, mode: str) -> None:
        with self._history_lock:
            self._history.appendleft(
                {
                    "id": self._next_entry_id,
                    "text": text,
                    "mode": mode,
                    "at": time.time(),
                }
            )
            self._next_entry_id += 1

    def history(self) -> list[dict]:
        """Newest first. Callers get copies so they cannot mutate the deque."""
        with self._history_lock:
            return [dict(entry) for entry in self._history]

    def clear_history(self) -> None:
        with self._history_lock:
            self._history.clear()
            self.last_text = None

    def _abandon(self) -> None:
        if self._recorder is not None:
            self._recorder.cancel()
        self._recording = False
        self._latched = False

    def _publish(self, phase: DictationPhase, **details) -> None:
        self.phase = phase
        self.last_error = details.get("error")
        if self._on_phase is None:
            return
        try:
            self._on_phase(phase, details)
        except Exception as error:
            log.error("dictation phase listener failed: %s", error_kind(error))
