"""Entry point for the installed app: double-click opens the web UI on the NPU.

All CLI flags still work (``oat-notes.exe --list-devices`` etc.); a bare
launch is rewritten to ``--ui --backend openvino --ov-device NPU``.
"""

import atexit
import os
import sys
from datetime import datetime
from pathlib import Path

from .paths import default_transcript_dir, log_dir


def _configure_windowed_logging() -> None:
    """A windowed build has no console: transcript-bearing stdout is discarded
    rather than persisted, while stderr diagnostics go to a dated log file."""
    if sys.stdout is None:
        sys.stdout = Path(os.devnull).open("w", encoding="utf-8")
        atexit.register(sys.stdout.close)
    if sys.stderr is None:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"oat-notes_{datetime.now():%Y-%m-%d}.log"
        sys.stderr = path.open("a", encoding="utf-8", buffering=1)
        atexit.register(sys.stderr.close)


_FROZEN_DEFAULTS = ("--ui", "--backend", "openvino", "--ov-device", "NPU")
_FLAGS_THAT_DO_NOT_SELECT_A_MODE = frozenset(
    {"--background", "--no-overlay", "--no-browser"}
)


def default_argv(argv: list[str]) -> list[str]:
    """A double-click, or the Windows startup shortcut, means 'run the app'.
    Anything that already picks a mode is left exactly as typed."""
    if any(
        argument not in _FLAGS_THAT_DO_NOT_SELECT_A_MODE for argument in argv[1:]
    ):
        return argv
    return [
        *argv,
        *_FROZEN_DEFAULTS,
        "--out-dir",
        str(default_transcript_dir()),
    ]


def run() -> None:
    _configure_windowed_logging()
    sys.argv = default_argv(sys.argv)
    from .cli import main

    main()
