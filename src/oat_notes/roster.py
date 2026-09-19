"""Rules for the live roster list: who sits on which hotkey slot, and how a
person is found by name, identity or slot."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from .attribution import Speaker

HOTKEYS_PER_BANK = 9


def assign_hotkey_slots(speakers: Iterable[Speaker]) -> list[Speaker]:
    """Honour requested slots; give everyone else the next free one."""
    roster: list[Speaker] = []
    used: set[int] = set()
    next_slot = 0
    for speaker in speakers:
        requested = speaker.hotkey_slot
        if requested is None or requested < 0 or requested in used:
            while next_slot in used:
                next_slot += 1
            requested = next_slot
        used.add(requested)
        next_slot = max(next_slot, requested + 1)
        roster.append(replace(speaker, hotkey_slot=requested))
    return roster


def next_free_slot(roster: Sequence[Speaker], bank: int) -> int:
    """The first empty slot in the current bank, else the first past everyone."""
    used = {speaker.hotkey_slot for speaker in roster}
    start = bank * HOTKEYS_PER_BANK
    for slot in range(start, start + HOTKEYS_PER_BANK):
        if slot not in used:
            return slot
    return max((slot for slot in used if slot is not None), default=-1) + 1


def index_for_slot(roster: Sequence[Speaker], slot: int) -> int | None:
    return next(
        (
            index
            for index, speaker in enumerate(roster)
            if speaker.hotkey_slot == slot and speaker.active
        ),
        None,
    )


def index_for(roster: Sequence[Speaker], name: str, speaker_id: str | None = None) -> int | None:
    """An active member with this saved identity, or failing that this name
    (case-insensitively)."""
    for index, speaker in enumerate(roster):
        if not speaker.active:
            continue
        same_identity = speaker_id is not None and speaker.speaker_id == speaker_id
        if same_identity or speaker.name.casefold() == name.casefold():
            return index
    return None


def check_index(roster: Sequence[Speaker], index: int) -> str | None:
    """Why ``index`` cannot be acted on, or None when it can."""
    if not 0 <= index < len(roster):
        return "speaker not found"
    if not roster[index].active:
        return "speaker already removed"
    return None
