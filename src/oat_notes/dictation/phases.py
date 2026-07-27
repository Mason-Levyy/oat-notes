"""The phases a dictation moves through.

Its own module because two unrelated consumers need the vocabulary: the
controller that publishes the phases, and the HUD that renders them. Neither
should have to import the other, and a HUD that imports the controller would
drag the whole capture and transcription stack in behind it.
"""

from __future__ import annotations

IDLE = "idle"
LISTENING = "listening"
LATCHED = "latched"
TRANSCRIBING = "transcribing"
FORMATTING = "formatting"
INSERTED = "inserted"
CANCELLED = "cancelled"
ERROR = "error"
LOADING = "loading"
