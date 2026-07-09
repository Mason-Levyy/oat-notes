"""Entry point for the packaged exe: double-click opens the web UI.

All CLI flags still work (``oat-notes.exe --list-devices`` etc.); a bare
launch is rewritten to ``--ui``.
"""

import sys

from .cli import main


def run() -> None:
    if len(sys.argv) == 1:
        sys.argv.append("--ui")
    main()
