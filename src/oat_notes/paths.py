"""Stable per-user locations used by the installed Windows application."""

from __future__ import annotations

import os
from pathlib import Path


def app_data_dir() -> Path:
    """Return the local, non-roaming application data directory."""
    root = os.environ.get("LOCALAPPDATA")
    if root:
        return Path(root) / "oat-notes"
    return Path.home() / ".oat-notes"


def documents_dir() -> Path:
    """Resolve the Windows Documents known folder, including OneDrive moves."""
    if os.name == "nt":
        try:
            import winreg

            key_path = (
                r"Software\Microsoft\Windows\CurrentVersion\Explorer"
                r"\User Shell Folders"
            )
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "Personal")
            return Path(os.path.expandvars(value))
        except (OSError, TypeError):
            pass
    return Path.home() / "Documents"


def default_transcript_dir() -> Path:
    return documents_dir() / "Oat Notes"


def log_dir() -> Path:
    return app_data_dir() / "logs"
