"""One running copy at a time.

Every instance installs a process-wide keyboard hook, so a second one arms on
the same chord: the same speech is recorded twice, transcribed twice, and
pasted twice. Binding the HTTP port used to be the only guard, and it only
catches a second instance that wants the same port — ``--port`` walks around
it.

A named kernel mutex is authoritative whatever the port. Windows destroys the
object when the last handle closes, including on a crash, so a stale lock
cannot lock the user out of their own app.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import wintypes
from pathlib import Path

from .paths import app_data_dir

MUTEX_NAME = "Local\\oat-notes-single-instance"
_ERROR_ALREADY_EXISTS = 183

_IS_WINDOWS = sys.platform == "win32"
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if _IS_WINDOWS else None

if _IS_WINDOWS:
    _kernel32.CreateMutexW.restype = wintypes.HANDLE
    _kernel32.CreateMutexW.argtypes = [
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


class SingleInstance:
    """Holds the lock for the life of the process."""

    def __init__(self, name: str = MUTEX_NAME, record: Path | None = None) -> None:
        self._name = name
        self._record = record or (app_data_dir() / "instance.json")
        self._handle: int | None = None

    def acquire(self, port: int) -> str | None:
        """``None`` when this process may run, otherwise the running instance's URL.

        The URL comes from what the running instance recorded, not from what
        this one was asked for, so launching a second copy on a different port
        still opens the UI that actually exists.
        """
        if not _IS_WINDOWS:
            return None
        handle = _kernel32.CreateMutexW(None, False, self._name)
        if not handle:
            return None
        if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
            _kernel32.CloseHandle(handle)
            return self._running_url(port)
        self._handle = handle
        self._remember(port)
        return None

    def release(self) -> None:
        if self._handle is not None:
            _kernel32.CloseHandle(self._handle)
            self._handle = None
        try:
            self._record.unlink(missing_ok=True)
        except OSError:
            pass

    def _running_url(self, fallback_port: int) -> str:
        port = fallback_port
        try:
            payload = json.loads(self._record.read_text(encoding="utf-8"))
            port = int(payload["port"])
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return f"http://127.0.0.1:{port}"

    def _remember(self, port: int) -> None:
        try:
            self._record.parent.mkdir(parents=True, exist_ok=True)
            self._record.write_text(
                json.dumps({"port": port, "pid": os.getpid()}) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
