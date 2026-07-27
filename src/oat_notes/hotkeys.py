"""Global hotkey chords for meeting speaker switching and dictation.

One process-wide pynput hook drives every binding. Chords match on an exact
modifier set. A chord with no key arms while its modifiers are held and
commits when they are released; pressing any other key cancels it, which is
what stops Windows' own Ctrl+Win+D and Ctrl+Win+arrow shortcuts from starting
a recording.

Callbacks run on the hook thread, which sits in the path of every keystroke
on the machine — keep them quick.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

MODIFIERS = ("ctrl", "alt", "shift", "win")

DIGIT = "digit"
BRACKET = "bracket"

_NUMPAD_ONE_VK = 97
_NUMPAD_NINE_VK = 105


@dataclass(frozen=True)
class Chord:
    """A modifier set plus an optional key: ``DIGIT``, ``BRACKET``, or a
    pynput key name such as ``"esc"``. No key means a modifier-only chord."""

    modifiers: frozenset[str]
    key: str | None = None

    @classmethod
    def of(cls, modifiers, key: str | None = None) -> Chord:
        requested = frozenset(modifiers)
        unknown = requested.difference(MODIFIERS)
        if unknown:
            raise ValueError(f"unknown hotkey modifier: {sorted(unknown)[0]}")
        if not requested and key is None:
            raise ValueError("a chord needs at least one modifier or a key")
        return cls(requested, key)

    @property
    def is_modifier_only(self) -> bool:
        return self.key is None


@dataclass
class _Binding:
    chord: Chord
    on_press: Callable
    on_release: Callable[[float], None] | None = None
    on_cancel: Callable[[], None] | None = None
    armed_at: float | None = None
    cancelled: bool = False


class HotkeyListener:
    """The process-wide hook. Constructing it with ``on_switch`` registers the
    meeting speaker-switch bindings directly."""

    def __init__(
        self,
        on_switch: Callable[[int], None] | None = None,
        modifiers: tuple[str, ...] = ("ctrl", "alt"),
        on_page: Callable[[int], None] = lambda delta: None,
    ) -> None:
        self._listener = None
        self._held: set[str] = set()
        self._bindings: dict[str, _Binding] = {}
        self._lock = threading.Lock()
        if on_switch is not None:
            if not modifiers or not set(modifiers).issubset(set(MODIFIERS)):
                raise ValueError(
                    "modifiers must contain ctrl, alt, shift, and/or win"
                )
            self.bind_speaker_switch(modifiers, on_switch, on_page)

    def start(self) -> None:
        if self._listener is not None:
            return
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._handle_press, on_release=self._handle_release
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._held.clear()

    def bind(
        self,
        name: str,
        chord: Chord,
        on_press: Callable,
        on_release: Callable[[float], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        """Register or replace a binding. ``on_release`` receives how long the
        chord was held, so the caller can tell a tap from a hold. Timed with
        perf_counter because monotonic() only ticks every ~15.6 ms on Windows,
        which is coarse against a 400 ms tap threshold."""
        with self._lock:
            self._bindings[name] = _Binding(chord, on_press, on_release, on_cancel)

    def unbind(self, name: str) -> None:
        with self._lock:
            self._bindings.pop(name, None)

    def bind_speaker_switch(
        self,
        modifiers,
        on_switch: Callable[[int], None],
        on_page: Callable[[int], None] = lambda delta: None,
    ) -> None:
        self.bind("speaker-switch", Chord.of(modifiers, DIGIT), on_switch)
        self.bind("speaker-page", Chord.of(modifiers, BRACKET), on_page)

    def unbind_speaker_switch(self) -> None:
        self.unbind("speaker-switch")
        self.unbind("speaker-page")

    def _snapshot(self) -> tuple[_Binding, ...]:
        """bind and unbind run on the HTTP thread while the hook thread is
        dispatching, so dispatch iterates a copy."""
        with self._lock:
            return tuple(self._bindings.values())

    def _handle_press(self, key) -> None:
        modifier = self._modifier_name(key)
        if modifier is not None:
            if modifier in self._held:
                return
            self._held.add(modifier)
            self._arm_modifier_chords()
            return
        self._cancel_armed()
        self._fire_key_chords(key)

    def _handle_release(self, key) -> None:
        modifier = self._modifier_name(key)
        if modifier is None:
            return
        self._held.discard(modifier)
        for binding in self._snapshot():
            if binding.armed_at is None:
                continue
            if self._held == binding.chord.modifiers:
                continue
            held_seconds = time.perf_counter() - binding.armed_at
            binding.armed_at = None
            if binding.cancelled:
                binding.cancelled = False
                continue
            if binding.on_release is not None:
                self._safely(binding.on_release, held_seconds)

    def _arm_modifier_chords(self) -> None:
        for binding in self._snapshot():
            if not binding.chord.is_modifier_only or binding.armed_at is not None:
                continue
            if self._held != binding.chord.modifiers:
                continue
            binding.armed_at = time.perf_counter()
            binding.cancelled = False
            self._safely(binding.on_press)

    def _cancel_armed(self) -> None:
        for binding in self._snapshot():
            if binding.armed_at is None or binding.cancelled:
                continue
            binding.cancelled = True
            if binding.on_cancel is not None:
                self._safely(binding.on_cancel)

    def _fire_key_chords(self, key) -> None:
        for binding in self._snapshot():
            chord = binding.chord
            if chord.is_modifier_only or self._held != chord.modifiers:
                continue
            matched, arguments = self._match_key(chord.key, key)
            if matched:
                self._safely(binding.on_press, *arguments)

    @classmethod
    def _match_key(cls, wanted: str | None, key) -> tuple[bool, tuple]:
        if wanted == DIGIT:
            index = cls._digit_index(key)
            return (False, ()) if index is None else (True, (index,))
        if wanted == BRACKET:
            direction = cls._page_direction(key)
            return (False, ()) if direction is None else (True, (direction,))
        return (getattr(key, "name", "") == wanted, ())

    @staticmethod
    def _safely(callback: Callable, *arguments) -> None:
        """An exception escaping a pynput callback kills the hook, taking every
        other binding with it."""
        try:
            callback(*arguments)
        except Exception as error:
            print(f"hotkey callback error: {type(error).__name__}", file=sys.stderr)

    @staticmethod
    def _modifier_name(key) -> str | None:
        name = getattr(key, "name", "")
        if name.startswith("ctrl"):
            return "ctrl"
        if name.startswith("alt"):
            return "alt"
        if name.startswith("shift"):
            return "shift"
        if name.startswith("cmd") or name.startswith("win"):
            return "win"
        return None

    @staticmethod
    def _digit_index(key) -> int | None:
        virtual_key = getattr(key, "vk", None)
        if virtual_key is not None and _NUMPAD_ONE_VK <= virtual_key <= _NUMPAD_NINE_VK:
            return virtual_key - _NUMPAD_ONE_VK
        char = getattr(key, "char", None)
        if char and char.isdigit() and char != "0":
            return int(char) - 1
        # Ctrl+Alt+digit on Windows often reports a control character
        # (\x11 for 1, etc.) or the raw vk instead of the digit char.
        if virtual_key is not None and ord("1") <= virtual_key <= ord("9"):
            return virtual_key - ord("1")
        return None

    @staticmethod
    def _page_direction(key) -> int | None:
        char = getattr(key, "char", None)
        if char == "[":
            return -1
        if char == "]":
            return 1
        virtual_key = getattr(key, "vk", None)
        if virtual_key == 219:  # OEM_4: [ on a US Windows keyboard
            return -1
        if virtual_key == 221:  # OEM_6: ] on a US Windows keyboard
            return 1
        return None
