"""Validated, persisted settings for the installed application."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import app_data_dir

log = logging.getLogger(__name__)

MODIFIER_ORDER = ("ctrl", "alt", "shift", "win")
MODIFIER_LABELS = {
    "ctrl": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "win": "Win",
}

DIGIT_KEY = "digit"
REPLAY_KEY = "z"
MIN_TAP_MS = 50
MAX_TAP_MS = 2000

WHISPER_MODELS = ("distil-small.en", "small.en", "medium.en")
WHISPER_MODEL_LABELS = {
    "distil-small.en": "fastest",
    "small.en": "balanced",
    "medium.en": "best on accents",
}


def normalize_modifiers(value: Any, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("hotkey_modifiers must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("hotkey modifiers must be strings")
    requested = {item.strip().lower() for item in value if item.strip()}
    unknown = requested.difference(MODIFIER_ORDER)
    if unknown:
        raise ValueError(f"unknown hotkey modifier: {min(unknown)}")
    if not requested and not allow_empty:
        raise ValueError("choose at least one hotkey modifier")
    return tuple(modifier for modifier in MODIFIER_ORDER if modifier in requested)


def modifier_label(modifiers: tuple[str, ...]) -> str:
    return "+".join(MODIFIER_LABELS[item] for item in modifiers)


def normalize_vocabulary(value: Any) -> tuple[tuple[str, str], ...]:
    """Pairs of (heard, written). Accepts the JSON shape the UI sends —
    a list of two-item lists — and drops entries with nothing to match."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("dictation_vocabulary must be a list")
    entries: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            heard, written = item.get("heard", ""), item.get("written", "")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            heard, written = item
        else:
            raise ValueError("each vocabulary entry needs a heard and written word")
        if not isinstance(heard, str) or not isinstance(written, str):
            raise ValueError("vocabulary entries must be strings")
        if not heard.strip():
            continue
        entries.append((heard.strip(), written.strip()))
    return tuple(entries)


def _choice(value: Any, allowed: tuple[str, ...], field: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise ValueError(f"{field} must be one of: {', '.join(allowed)}")
    return value.strip().lower()


def _flag(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _chords_collide(
    one: tuple[tuple[str, ...], str | None],
    other: tuple[tuple[str, ...], str | None],
) -> bool:
    """Same modifiers and same key collide; so does any pair where either side
    is modifier-only, since it fires on the modifiers alone."""
    (modifiers, key), (other_modifiers, other_key) = one, other
    if modifiers != other_modifiers:
        return False
    return key is None or other_key is None or key == other_key


@dataclass(frozen=True)
class AppSettings:
    hotkey_modifiers: tuple[str, ...] = ("ctrl", "alt")
    whisper_model: str = "small.en"
    dictation_enabled: bool = True
    dictation_modifiers: tuple[str, ...] = ("ctrl", "win")
    dictation_email_modifiers: tuple[str, ...] = ("ctrl", "shift", "win")
    dictation_replay_modifiers: tuple[str, ...] = ("ctrl", "alt")
    dictation_tap_ms: int = 400
    dictation_email_detection: bool = True
    dictation_restore_clipboard: bool = True
    dictation_spoken_punctuation: bool = False
    dictation_vocabulary: tuple[tuple[str, str], ...] = ()
    overlay_enabled: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AppSettings:
        if not isinstance(payload, dict):
            raise ValueError("settings must be an object")
        defaults = cls()

        def pick(name: str):
            return payload.get(name, getattr(defaults, name))

        tap_ms = pick("dictation_tap_ms")
        if not isinstance(tap_ms, int) or isinstance(tap_ms, bool):
            raise ValueError("dictation_tap_ms must be a whole number of milliseconds")
        if not MIN_TAP_MS <= tap_ms <= MAX_TAP_MS:
            raise ValueError(
                f"dictation_tap_ms must be between {MIN_TAP_MS} and {MAX_TAP_MS}"
            )
        settings = cls(
            hotkey_modifiers=normalize_modifiers(pick("hotkey_modifiers")),
            whisper_model=_choice(
                pick("whisper_model"), WHISPER_MODELS, "whisper_model"
            ),
            dictation_enabled=_flag(pick("dictation_enabled"), "dictation_enabled"),
            dictation_modifiers=normalize_modifiers(pick("dictation_modifiers")),
            dictation_email_modifiers=normalize_modifiers(
                pick("dictation_email_modifiers"), allow_empty=True
            ),
            dictation_replay_modifiers=normalize_modifiers(
                pick("dictation_replay_modifiers"), allow_empty=True
            ),
            dictation_tap_ms=tap_ms,
            dictation_email_detection=_flag(
                pick("dictation_email_detection"), "dictation_email_detection"
            ),
            dictation_restore_clipboard=_flag(
                pick("dictation_restore_clipboard"), "dictation_restore_clipboard"
            ),
            dictation_spoken_punctuation=_flag(
                pick("dictation_spoken_punctuation"), "dictation_spoken_punctuation"
            ),
            dictation_vocabulary=normalize_vocabulary(pick("dictation_vocabulary")),
            overlay_enabled=_flag(pick("overlay_enabled"), "overlay_enabled"),
        )
        settings._check_chords()
        return settings

    def _check_chords(self) -> None:
        """Two bindings that resolve to the same keystroke make one of them
        unreachable.

        A chord is its modifiers *plus* its key, so Ctrl+Alt+Z and Ctrl+Alt+1–9
        coexist happily. Modifier-only chords are the exception: they arm on the
        modifiers alone, so they may not share modifiers with anything.
        """
        chords = [
            ("the speaker-switch hotkey", self.hotkey_modifiers, DIGIT_KEY),
            ("the dictation hotkey", self.dictation_modifiers, None),
        ]
        if self.dictation_email_modifiers:
            chords.append(
                ("the email dictation hotkey", self.dictation_email_modifiers, None)
            )
        if self.dictation_replay_modifiers:
            chords.append(
                ("the replay hotkey", self.dictation_replay_modifiers, REPLAY_KEY)
            )

        for index, (name, modifiers, key) in enumerate(chords):
            for other_name, other_modifiers, other_key in chords[:index]:
                if not _chords_collide((modifiers, key), (other_modifiers, other_key)):
                    continue
                raise ValueError(
                    f"{name} and {other_name} would both be"
                    f" {modifier_label(modifiers)} — choose different modifiers"
                )

    def merged(self, payload: Mapping[str, Any]) -> AppSettings:
        """Apply a partial update. The settings page saves one card at a time,
        so a request that omits a field must leave it alone rather than reset
        it to the default."""
        if not isinstance(payload, dict):
            raise ValueError("settings must be an object")
        return AppSettings.from_dict({**self.to_dict(), **payload})

    @property
    def hotkey_label(self) -> str:
        return f"{modifier_label(self.hotkey_modifiers)}+1–9"

    @property
    def dictation_label(self) -> str:
        return modifier_label(self.dictation_modifiers)

    @property
    def dictation_email_label(self) -> str:
        if not self.dictation_email_modifiers:
            return "off"
        return modifier_label(self.dictation_email_modifiers)

    @property
    def dictation_replay_label(self) -> str:
        if not self.dictation_replay_modifiers:
            return "off"
        return f"{modifier_label(self.dictation_replay_modifiers)}+{REPLAY_KEY.upper()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotkey_modifiers": list(self.hotkey_modifiers),
            "hotkey_label": self.hotkey_label,
            "whisper_model": self.whisper_model,
            "whisper_models": [
                {"name": name, "label": WHISPER_MODEL_LABELS[name]}
                for name in WHISPER_MODELS
            ],
            "dictation_enabled": self.dictation_enabled,
            "dictation_modifiers": list(self.dictation_modifiers),
            "dictation_label": self.dictation_label,
            "dictation_email_modifiers": list(self.dictation_email_modifiers),
            "dictation_email_label": self.dictation_email_label,
            "dictation_replay_modifiers": list(self.dictation_replay_modifiers),
            "dictation_replay_label": self.dictation_replay_label,
            "dictation_tap_ms": self.dictation_tap_ms,
            "dictation_email_detection": self.dictation_email_detection,
            "dictation_restore_clipboard": self.dictation_restore_clipboard,
            "dictation_spoken_punctuation": self.dictation_spoken_punctuation,
            "dictation_vocabulary": [
                [heard, written] for heard, written in self.dictation_vocabulary
            ],
            "overlay_enabled": self.overlay_enabled,
        }


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (app_data_dir() / "settings.json")

    def load(self) -> AppSettings:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return AppSettings.from_dict(payload)
        except FileNotFoundError:
            return AppSettings()
        except (OSError, ValueError, TypeError) as error:
            log.warning("ignoring invalid settings file: %s", error)
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(settings.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
