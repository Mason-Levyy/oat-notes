"""Collapse the phrase-level repetition loops Whisper falls into.

Greedy decoding can lock onto a phrase and emit it until the chunk runs out.
The decoder-side guards in ``transcriber`` catch most of it; this is the
backstop for what still gets through, and the only guard on a loop that
straddles two chunks — dictation joins several chunks into one utterance, so
no single decode ever sees the repetition.

``format._REPEAT_PATTERN`` already collapses a stuttered *word* and owns the
list of words that legitimately double. This deliberately starts where that
stops: a single word has to appear three times before it is treated as a loop.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\S+")
_STRIPPED = "\"'“”‘’.,!?;:—–-…()[]"
# A loop is a short phrase said over and over. Bounding the window keeps the
# scan linear enough on a long dictation and cannot miss a real one.
_MAX_RUN_TOKENS = 60


def _key(token: str) -> str:
    """Match on the words alone, so a run still collapses when only the last
    repetition carries the full stop."""
    stripped = token.strip(_STRIPPED)
    return (stripped or token).casefold()


def _is_a_loop(run_length: int, repeats: int) -> bool:
    """Three or more for one- and two-word runs, which are said twice often
    enough in real speech ("no, no", "thank you, thank you"). A phrase of
    three or more words repeated back to back is already a loop."""
    if run_length >= 3:
        return repeats >= 2
    return repeats >= 3


def _repeats_at(keys: list[str], start: int, run_length: int) -> int:
    repeats = 1
    while (
        start + (repeats + 1) * run_length <= len(keys)
        and keys[start + repeats * run_length : start + (repeats + 1) * run_length]
        == keys[start : start + run_length]
    ):
        repeats += 1
    return repeats


def _collapse_line(line: str) -> str:
    tokens = list(_TOKEN.finditer(line))
    keys = [_key(token.group()) for token in tokens]
    total = len(keys)

    cuts: list[tuple[int, int]] = []
    position = 0
    while position < total:
        longest = min(_MAX_RUN_TOKENS, (total - position) // 2)
        best_length = 0
        best_repeats = 0
        # Ascending, keeping only a strict improvement: "A B. A B. A B."
        # matches at both three and six tokens, and three is the real period.
        for run_length in range(1, longest + 1):
            repeats = _repeats_at(keys, position, run_length)
            if not _is_a_loop(run_length, repeats):
                continue
            if run_length * repeats > best_length * best_repeats:
                best_length, best_repeats = run_length, repeats
        if not best_length:
            position += 1
            continue
        # Cut from the end of the last kept token so the separator goes too.
        cuts.append(
            (
                tokens[position + best_length - 1].end(),
                tokens[position + best_length * best_repeats - 1].end(),
            )
        )
        position += best_length * best_repeats

    for start, end in reversed(cuts):
        line = line[:start] + line[end:]
    return line


def collapse_repeated_runs(text: str) -> str:
    """Reduce every back-to-back repetition of a token run to one occurrence.

    The first occurrence is kept verbatim and everything outside a collapsed
    run is returned byte for byte, so this is safe to run over text nothing
    else is going to normalize.
    """
    if not text:
        return text
    return "\n".join(_collapse_line(line) for line in text.split("\n"))
