"""Validated, persisted settings for the installed application."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import app_data_dir

MODIFIER_ORDER = ("ctrl", "alt", "shift", "win")
MODIFIER_LABELS = {
    "ctrl": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "win": "Win",
}


def normalize_modifiers(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("hotkey_modifiers must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("hotkey modifiers must be strings")
    requested = {item.strip().lower() for item in value}
    unknown = requested.difference(MODIFIER_ORDER)
    if unknown:
        raise ValueError(f"unknown hotkey modifier: {sorted(unknown)[0]}")
    if not requested:
        raise ValueError("choose at least one hotkey modifier")
    return tuple(modifier for modifier in MODIFIER_ORDER if modifier in requested)


@dataclass(frozen=True)
class AppSettings:
    hotkey_modifiers: tuple[str, ...] = ("ctrl", "alt")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AppSettings:
        if not isinstance(payload, dict):
            raise ValueError("settings must be an object")
        return cls(
            hotkey_modifiers=normalize_modifiers(
                payload.get("hotkey_modifiers", cls().hotkey_modifiers)
            )
        )

    @property
    def hotkey_label(self) -> str:
        prefix = "+".join(MODIFIER_LABELS[item] for item in self.hotkey_modifiers)
        return f"{prefix}+1–9"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotkey_modifiers": list(self.hotkey_modifiers),
            "hotkey_label": self.hotkey_label,
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
            print(f"warning: ignoring invalid settings file: {error}", file=sys.stderr)
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(settings.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
