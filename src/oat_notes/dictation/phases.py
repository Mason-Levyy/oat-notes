"""The phases a dictation moves through.

Its own module because two unrelated consumers need the vocabulary: the
controller that publishes the phases, and the HUD that renders them. Neither
should have to import the other, and a HUD that imports the controller would
drag the whole capture and transcription stack in behind it.
"""

from __future__ import annotations

from typing import Final, Literal

DictationPhase = Literal[
    "idle",
    "listening",
    "latched",
    "transcribing",
    "formatting",
    "inserted",
    "cancelled",
    "error",
    "loading",
]

IDLE: Final = "idle"
LISTENING: Final = "listening"
LATCHED: Final = "latched"
TRANSCRIBING: Final = "transcribing"
FORMATTING: Final = "formatting"
INSERTED: Final = "inserted"
CANCELLED: Final = "cancelled"
ERROR: Final = "error"
LOADING: Final = "loading"
