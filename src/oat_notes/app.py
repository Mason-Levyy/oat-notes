"""Entry point for the installed app: double-click opens the web UI on the NPU.

All CLI flags still work (``oat-notes.exe --list-devices`` etc.); a bare
launch is rewritten to ``--ui --backend openvino --ov-device NPU``.
"""

import atexit
import os
import sys
from datetime import datetime
from typing import TextIO

from .paths import default_transcript_dir, log_dir

_log_file: TextIO | None = None
_stdout_sink: TextIO | None = None


def _configure_windowed_logging() -> None:
    """Keep diagnostics, but never persist transcript-bearing stdout."""
    global _log_file, _stdout_sink
    if sys.stdout is not None and sys.stderr is not None:
        return
    if sys.stdout is None:
        # CLI transcript lines are written to stdout. A windowed build has no
        # console, so discard them instead of redirecting captured content to
        # the application's durable diagnostic log.
        _stdout_sink = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = _stdout_sink
    if sys.stderr is None:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"oat-notes_{datetime.now():%Y-%m-%d}.log"
        _log_file = path.open("a", encoding="utf-8", buffering=1)
        sys.stderr = _log_file
    if _stdout_sink is not None:
        atexit.register(_stdout_sink.close)
    if _log_file is not None:
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
