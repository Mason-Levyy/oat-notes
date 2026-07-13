"""Entry point for the installed app: double-click opens the web UI on the NPU.

All CLI flags still work (``oat-notes.exe --list-devices`` etc.); a bare
launch is rewritten to ``--ui --backend openvino --ov-device NPU``.
"""

import atexit
import sys
from datetime import datetime
from typing import TextIO

from .paths import default_transcript_dir, log_dir

_log_file: TextIO | None = None


def _configure_windowed_logging() -> None:
    """Give windowed PyInstaller builds a durable stdout/stderr target."""
    global _log_file
    if sys.stdout is not None and sys.stderr is not None:
        return
    directory = log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"oat-notes_{datetime.now():%Y-%m-%d}.log"
    _log_file = path.open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _log_file
    if sys.stderr is None:
        sys.stderr = _log_file
    atexit.register(_log_file.close)


def run() -> None:
    _configure_windowed_logging()
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--ui",
                "--backend",
                "openvino",
                "--ov-device",
                "NPU",
                "--out-dir",
                str(default_transcript_dir()),
            ]
        )
    from .cli import main

    main()
