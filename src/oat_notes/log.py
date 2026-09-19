"""One stderr log for the whole application.

``app.py`` points stderr at a dated file in the windowed build, so
``configure`` must run after that redirect."""

from __future__ import annotations

import logging
import sys

ROOT = "oat_notes"


def configure(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(threadName)s %(name)s %(levelname)s: %(message)s")
    )
    root = logging.getLogger(ROOT)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def error_kind(error: BaseException) -> str:
    """The exception's type, never its message: transcription, cleanup and
    dictation backends can echo captured speech into an error message, and
    the application log is durable."""
    return type(error).__name__
