"""Entry point for the packaged exe: double-click opens the web UI on the NPU.

All CLI flags still work (``oat-notes.exe --list-devices`` etc.); a bare
launch is rewritten to ``--ui --backend openvino --ov-device NPU``.
"""

import sys

from .cli import main


def run() -> None:
    if len(sys.argv) == 1:
        sys.argv.extend(["--ui", "--backend", "openvino", "--ov-device", "NPU"])
    main()
