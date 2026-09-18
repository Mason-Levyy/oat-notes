"""Joining the transcribed fragments of one utterance back into prose.

Whisper sees each VAD chunk on its own, so every fragment arrives looking
like a complete sentence: capital first word, full stop at the end. The pause
that produced the split says which of those were real. A breath is not a
sentence boundary; a long silence is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SELF_CAPITALISED = re.compile(r"^I(?:'(?:m|ll|ve|d)|$|\s)")
_MIN_WORDS_FOR_FULL_STOP = 3


@dataclass(frozen=True)
class Piece:
    """``pause_before`` is the silence between the speech that ended the
    previous fragment and the speech that opened this one — zero for the
    first fragment and after a forced split."""

    text: str
    pause_before: float = 0.0


def stitch_pieces(
    pieces: list[Piece], clause_gap_seconds: float, sentence_gap_seconds: float
) -> str:
    stitched = ""
    for piece in pieces:
        text = piece.text.strip()
        if not text:
            continue
        if not stitched:
            stitched = text
        elif piece.pause_before < clause_gap_seconds:
            stitched = _continue_sentence(stitched, text)
        elif piece.pause_before >= sentence_gap_seconds:
            stitched = _end_sentence(stitched, text)
        else:
            stitched = f"{stitched} {text}"
    return stitched


def _continue_sentence(stitched: str, text: str) -> str:
    trailing_off = stitched.endswith("...")
    if stitched.endswith(".") and not trailing_off:
        stitched = stitched[:-1].rstrip()
    if trailing_off or stitched[-1] not in "!?":
        text = _lowercase_first_word(text)
    return f"{stitched} {text}"


def _end_sentence(stitched: str, text: str) -> str:
    is_sentence_length = len(stitched.split()) >= _MIN_WORDS_FOR_FULL_STOP
    if stitched[-1].isalnum() and is_sentence_length:
        stitched += "."
    return f"{stitched} {text[0].upper()}{text[1:]}"


def _lowercase_first_word(text: str) -> str:
    first_word = text.split(maxsplit=1)[0]
    is_acronym = len(first_word) > 1 and first_word.isupper()
    if is_acronym or _SELF_CAPITALISED.match(text):
        return text
    return text[0].lower() + text[1:]
